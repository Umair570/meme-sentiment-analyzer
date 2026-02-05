import pandas as pd  # Import Pandas for handling data tables (DataFrames)
import numpy as np  # Import NumPy for efficient math and array operations
import os  # Import OS tools to interact with the file system (check files, make folders)
import re  # Import Regular Expressions for advanced text cleaning (finding patterns)
import requests  # Import requests to download images from the internet
import pickle  # Import pickle to save trained models and objects to files
import matplotlib.pyplot as plt  # Import Matplotlib to draw graphs (like confusion matrices)
from PIL import Image  # Import Python Imaging Library to open and process images
from io import BytesIO  # Import BytesIO to handle image data in memory (RAM) without saving to disk
from tqdm import tqdm  # Import tqdm to display progress bars during long loops
import concurrent.futures  # Import tools for Multithreading (downloading multiple images at once)
from scipy import sparse  # Import sparse matrices to save memory when data has lots of zeros

# Scikit-Learn & Imbalance
from sklearn.feature_extraction.text import TfidfVectorizer  # Tool to convert text into statistical numbers
from sklearn.linear_model import SGDClassifier   # Import the Stochastic Gradient Descent classifier model
from sklearn.ensemble import RandomForestClassifier  # Import the Random Forest classifier model
from sklearn.model_selection import train_test_split  # Tool to split data into Training and Validation sets
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay  # Tools to grade model performance
from sklearn.kernel_approximation import Nystroem # Import Nyström approximation for creating non-linear features

# TensorFlow (Feature Extraction Only)
from tensorflow.keras.applications import ResNet50  # Import the pre-trained ResNet50 Deep Learning model
from tensorflow.keras.applications.resnet50 import preprocess_input  # Import math function to clean images for ResNet
from tensorflow.keras.utils import img_to_array  # Tool to convert an Image object into a NumPy number array

# ---------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------
CSV_FILE = "train_cleaned_final.csv"  # Define the name of the dataset file
IMG_CACHE_FILE = "image_features_resnet.npy"  # Define the file name to store processed image features (cache)
MODELS_DIR = "models"  # Define the folder name where trained models will be saved
MATRIX_DIR = "confusion_matrices"  # Define the folder name where performance graphs will be saved

os.makedirs(MODELS_DIR, exist_ok=True)  # Create the models folder if it doesn't exist
os.makedirs(MATRIX_DIR, exist_ok=True)  # Create the matrices folder if it doesn't exist

# ---------------------------------------------------------
# 2. LOAD DATA & CLEAN
# ---------------------------------------------------------
# Check if the CSV file exists before proceeding
if not os.path.exists(CSV_FILE):
    print(f"❌ Error: {CSV_FILE} not found.")  # Print error if missing
    exit()  # Stop the program immediately

print("🔹 Loading dataset...")
try:
    df = pd.read_csv(CSV_FILE, encoding='utf-8')  # Try loading CSV with standard UTF-8 encoding
except:
    df = pd.read_csv(CSV_FILE, encoding='latin1')  # Fallback to latin1 encoding if UTF-8 fails

# Clean column names: make lowercase and remove spaces
df.columns = df.columns.str.lower().str.strip()

# Clean OCR Text column
df['ocr'] = df['ocr'].fillna("").astype(str).str.lower()  # Fill missing text, convert to string, make lowercase
# Remove all characters that are NOT letters (a-z), numbers (0-9), or spaces
df['ocr'] = df['ocr'].apply(lambda x: re.sub(r'[^a-z0-9\s]', '', x))

print(f"✅ Dataset loaded. Total Rows: {len(df)}")  # Print total number of rows loaded

# Define a function to merge similar labels (e.g., "hilarious" -> "funny")
def merge_labels(label_col, merge_map, default):
    if label_col not in df.columns: return  # specific column doesn't exist, skip
    # Map the old labels to new ones using the dictionary; use default if not found
    df[label_col] = df[label_col].map(merge_map).fillna(default).astype(str)

# Apply the merging logic to the 'humour' column
merge_labels("humour", {"funny": "funny", "hilarious": "funny", "not_funny": "not_funny", "sarcastic": "sarcastic"}, "not_funny")
# Apply the merging logic to the 'sarcastic' column
merge_labels("sarcastic", {"sarcastic": "sarcastic", "not_sarcastic": "not_sarcastic", "general": "sarcastic", "twisted_meaning": "sarcastic"}, "not_sarcastic")
# Apply the merging logic to the 'offensive' column
merge_labels("offensive", {"offensive": "offensive", "hateful_offensive": "offensive", "slightly": "offensive", "not_offensive": "not_offensive"}, "not_offensive")
# Apply the merging logic to the 'motivational' column
merge_labels("motivational", {"motivational": "motivational", "not_motivational": "not_motivational"}, "not_motivational")
# Apply the merging logic to the 'overall' column
merge_labels("overall", {"positive": "positive", "negative": "negative", "neutral": "neutral", "very_positive": "positive"}, "neutral")

# ---------------------------------------------------------
# 3. IMAGE FEATURE EXTRACTION (WITH CACHING)
# ---------------------------------------------------------
# Check if we have already processed the images and saved them to disk
if os.path.exists(IMG_CACHE_FILE):
    print(f"✅ Found cached image features! Loading {IMG_CACHE_FILE}...")
    img_features = np.load(IMG_CACHE_FILE)  # Load the features from the file
    
    # Safety check: Ensure the number of loaded features matches the CSV rows
    if img_features.shape[0] != len(df):
        print(f"⚠️ Shape mismatch! Cache: {img_features.shape[0]}, CSV: {len(df)}")
        print("⚠️ Deleting cache and re-running extraction...")
        os.remove(IMG_CACHE_FILE)  # Delete the bad cache file
        img_features = None  # Reset variable to trigger re-extraction
else:
    img_features = None  # No file found

# If features are missing, start the extraction process
if img_features is None:
    print("\n🔹 No valid cache found. Starting Extraction...")
    # Initialize ResNet50 model, excluding the top classification layer (include_top=False)
    resnet = ResNet50(weights='imagenet', include_top=False, pooling='avg')

    # Define helper function to download and process one image
    def fetch_process_image(url):
        try:
            response = requests.get(url, timeout=5)  # Download image with 5s timeout
            if response.status_code == 200:  # If download successful
                img = Image.open(BytesIO(response.content))  # Open image from memory
                if img.mode != 'RGB': img = img.convert('RGB')  # Ensure image is color (RGB)
                img = img.resize((224, 224))  # Resize to 224x224 (ResNet requirement)
                x = img_to_array(img)  # Convert image to numbers (array)
                x = np.expand_dims(x, axis=0)  # Add batch dimension (1, 224, 224, 3)
                x = preprocess_input(x)  # Apply ResNet specific color cleaning
                return x  # Return processed image
        except:
            pass  # Ignore errors (bad links)
        return None  # Return Nothing if failed

    urls = df['image_url'].tolist()  # Get list of all URLs
    batch_size = 64  # Process 64 images at a time
    img_features_list = []  # List to store results
    
    # Start progress bar loop
    with tqdm(total=len(urls)) as pbar:
        for i in range(0, len(urls), batch_size):
            batch_urls = urls[i : i + batch_size]  # Get current batch of URLs
            
            # Use ThreadPoolExecutor to download 20 images simultaneously
            with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
                results = list(executor.map(fetch_process_image, batch_urls))
            
            valid_imgs = []  # List for successful images
            valid_indices = []  # List for their original positions
            batch_feats = np.zeros((len(batch_urls), 2048))  # Placeholder for features
            
            # Filter out failed downloads
            for idx, img_data in enumerate(results):
                if img_data is not None:
                    valid_imgs.append(img_data)
                    valid_indices.append(idx)
            
            # If we have valid images, extract features using ResNet
            if valid_imgs:
                valid_batch = np.vstack(valid_imgs)  # Stack images into one block
                preds = resnet.predict(valid_batch, verbose=0)  # Get AI features
                for v_idx, pred in zip(valid_indices, preds):
                    batch_feats[v_idx] = pred  # Store features in correct spot
            
            img_features_list.append(batch_feats)  # Add batch features to main list
            pbar.update(len(batch_urls))  # Update progress bar

    img_features = np.vstack(img_features_list)  # Stack all batches into final array
    np.save(IMG_CACHE_FILE, img_features)  # Save array to disk for next time
    print(f"✅ Features extracted and saved to {IMG_CACHE_FILE}")

# ---------------------------------------------------------
# 4. TEXT & COMBINE (MAX ACCURACY FEATURE ENGINEERING)
# ---------------------------------------------------------
print("🔹 Vectorizing text (Max Accuracy Feature Engineering: Unigrams & Bigrams)...")

# Initialize TF-IDF Vectorizer: keep top 4000 words, remove English stop words
# ngram_range=(1, 2) means use single words AND pairs of words (Bigrams)
vectorizer = TfidfVectorizer(max_features=4000, stop_words='english', ngram_range=(1, 2))
X_text = vectorizer.fit_transform(df['ocr'])  # Learn words and convert text to numbers

# Save the vectorizer settings to a file so app.py can use it
with open(f"{MODELS_DIR}/vectorizer.pkl", 'wb') as f:
    pickle.dump(vectorizer, f)

X_img_sparse = sparse.csr_matrix(img_features)  # Compress image features to save RAM
X_combined = sparse.hstack([X_text, X_img_sparse])  # Combine Text features + Image features side-by-side

# ---------------------------------------------------------
# 5. TRAIN MODELS (NON-LINEAR SGD WITH NYSTRÖM)
# ---------------------------------------------------------
labels = ["humour", "sarcastic", "offensive", "motivational", "overall"]  # List of target labels

print("\n🚀 Starting Training (Nyström Non-Linear SGD)...")

# Base parameters for the SGD Classifier (Fast linear model)
SGD_PARAMS = {
    "loss": 'modified_huber',   # Loss function suited for probability
    "class_weight": "balanced", # Handle uneven data (e.g., less offensive memes)
    "max_iter": 1000,           # Maximum number of passes over data
    "tol": 1e-3,                # Stopping criteria
    "penalty": 'l2',            # Regularization to prevent overfitting
    "alpha": 0.0001,            # Regularization strength
    "n_jobs": -1,               # Use all CPU cores
    "random_state": 42          # Seed for reproducibility
}

# Initialize Nystroem transformer for Kernel Approximation
# This creates 500 synthetic non-linear features to help linear models learn complex patterns
nystroem = Nystroem(gamma=0.1, n_components=500, random_state=42)
NYSTROEM_SAVED = False # Flag to ensure we only save the transformer once

# Loop through each label to train a specific model for it
for label in labels:
    if label not in df.columns: continue  # Skip if label missing from CSV

    print(f"\n--- Processing label: {label} ---")
    y = df[label].astype(str)  # Get targets for this label
    
    # Split data: 80% Training, 20% Validation (Testing)
    X_train, X_val, y_train, y_val = train_test_split(
        X_combined, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 💥 CRITICAL STEP: Apply Nystroem transformation for non-linear features
    print("   Applying Nystroem Transform...")
    
    # Learn the pattern on Training data and transform it
    X_train_transformed = nystroem.fit_transform(X_train)
    
    # Save the Nystroem transformer only once (used for all labels except overall)
    if not NYSTROEM_SAVED:
        print("   Saving fitted Nystroem Transformer...")
        with open(f"{MODELS_DIR}/nystroem_transformer.pkl", 'wb') as f:
            pickle.dump(nystroem, f)
        NYSTROEM_SAVED = True
    
    # Apply the learned transformation to the Validation data
    X_val_transformed = nystroem.transform(X_val)
    
    # Combine Original Features + New Non-Linear Features
    X_train_final = sparse.hstack([X_train, sparse.csr_matrix(X_train_transformed)]).tocsr()
    X_val_final = sparse.hstack([X_val, sparse.csr_matrix(X_val_transformed)]).tocsr()
    
    # Model Selection Logic based on label difficulty
    if label == 'overall':
        # Use Random Forest for Overall sentiment (Handles general classification well)
        model = RandomForestClassifier(class_weight="balanced", random_state=42, n_estimators=200, max_depth=30, n_jobs=-1)
        X_train_fit = X_train  # RF doesn't need Nystroem features
        X_val_fit = X_val
    
    elif label in ['offensive', 'sarcastic']:
        # Use SGD with extreme tuning for difficult classes
        tuned_params = SGD_PARAMS.copy()
        tuned_params['alpha'] = 1e-7 # Very low alpha for higher sensitivity
        tuned_params['penalty'] = 'elasticnet' # Mix of L1/L2 regularization
        model = SGDClassifier(**tuned_params)
        X_train_fit = X_train_final # Use Nystroem features
        X_val_fit = X_val_final
        
    else:
        # Use Standard SGD with Nystroem for other labels (Humour, Motivational)
        model = SGDClassifier(**SGD_PARAMS)
        X_train_fit = X_train_final # Use Nystroem features
        X_val_fit = X_val_final


    print(f"Training {model.__class__.__name__} for '{label}'...")
    model.fit(X_train_fit, y_train)  # TRAIN the model
    print("Training complete.")

    # Evaluate the model
    y_pred = model.predict(X_val_fit)  # Predict on validation set
    print(classification_report(y_val, y_pred, zero_division=0))  # Print accuracy report

    # Save Artifacts (Confusion Matrix and Model)
    classes = model.classes_

    cm = confusion_matrix(y_val, y_pred, labels=classes)  # Calculate confusion matrix
    # Plot and save the confusion matrix image
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes).plot(cmap='Blues', xticks_rotation='vertical')
    plt.title(f'Confusion Matrix: {label}')
    plt.savefig(f"{MATRIX_DIR}/{label}_cm.png", bbox_inches='tight')
    plt.close()
    
    # Save the trained model object to a file
    with open(f"{MODELS_DIR}/{label}_model.pkl", 'wb') as f:
        pickle.dump(model, f)

print("\n🎉 Training complete!")