from flask import Flask, render_template, request, url_for
import os, re, pickle, numpy as np
from PIL import Image
from scipy import sparse
import pytesseract

# TensorFlow / Keras
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import img_to_array

# Scikit-Learn
# Importing these is required for pickle to reconstruct the model objects
from sklearn.linear_model import SGDClassifier 
from sklearn.kernel_approximation import Nystroem 
from sklearn.ensemble import RandomForestClassifier

# ------------------------------
# Flask Configuration
# ------------------------------
app = Flask(__name__)
app.config["IMAGE_UPLOADS"] = os.path.join("static", "uploads")
os.makedirs(app.config["IMAGE_UPLOADS"], exist_ok=True)

# Optional: set Tesseract path (Windows only)
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Global variables
NYSTROEM_TRANSFORMER = None
VECTORIZER = None

# ------------------------------
# Load Models & Vectorizer
# ------------------------------
print("🔹 Loading vectorizer, Nystroem transformer, and models...")

try:
    # Load Vectorizer
    VECTORIZER = pickle.load(open("models/vectorizer.pkl", "rb"))
    
    # Load Nystroem Transformer (Required for SGD models)
    try:
        NYSTROEM_TRANSFORMER = pickle.load(open("models/nystroem_transformer.pkl", "rb"))
        print("✅ Nystroem Transformer loaded successfully.")
    except FileNotFoundError:
        print("⚠️ Warning: models/nystroem_transformer.pkl not found. SGD predictions may fail due to feature mismatch.")

except FileNotFoundError as e:
    print(f"❌ Error: Required preprocessing files (vectorizer.pkl) not found. Run train.py first. Details: {e}")
    exit()

# Load ResNet50 (Must match train.py config: pooling='avg')
resnet = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# Friendly label mapping
label_mapping = {
    "humour": {
        "funny": "Humorous", "not_funny": "Not Humorous", "sarcastic": "Sarcastic"
    },
    "sarcastic": {
        "sarcastic": "Sarcastic", "not_sarcastic": "Not Sarcastic" 
    },
    "offensive": {
        "offensive": "Offensive", "not_offensive": "Not Offensive"
    },
    "motivational": {
        "motivational": "Motivational", "not_motivational": "Not Motivational"
    },
    "overall": {
        "positive": "Positive", "negative": "Negative", "neutral": "Neutral"
    }
}

# ------------------------------
# Helper Functions
# ------------------------------
def extract_image_features(img_path):
    """Extract CNN features using ResNet50."""
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        
        features = resnet.predict(x, verbose=0)
        return features.flatten()
    except Exception as e:
        print(f"[ERROR] Image feature extraction failed: {e}")
        # Return zeros if image fails, matching dimension (2048)
        return np.zeros(2048)

def process_image(image_path):
    """
    Extract text, clean, vectorize, extract image features, and combine (X_original).
    """
    try:
        img_pil = Image.open(os.path.abspath(image_path))
        text = pytesseract.image_to_string(img_pil)
        print(f"📝 OCR Text Extracted: {text[:120]}...")
    except Exception as e:
        print(f"[ERROR] OCR failed: {e}")
        text = ""

    # Clean text (Exact match to train.py logic)
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    
    # Vectorize text
    X_text = VECTORIZER.transform([text])

    # Extract image features
    X_img = extract_image_features(image_path)

    # Combine text + image features (Original features)
    # train.py: X_combined = sparse.hstack([X_text, X_img_sparse])
    X_combined_original = sparse.hstack([X_text, sparse.csr_matrix(X_img)]).tocsr()
    
    return X_combined_original

def prepare_for_prediction(X_original, label_name, nystroem_transformer):
    """
    Applies the necessary Nystroem transformation based on the model training logic.
    """
    # 1. 'overall' was trained on RandomForest with Original Features ONLY
    if label_name == 'overall':
        return X_original
    
    # 2. All other labels (humour, sarcastic, etc.) were trained on SGD with Nystroem Features
    elif nystroem_transformer:
        # Apply transformation
        X_transformed = nystroem_transformer.transform(X_original)
        
        # Stack: [Original, Nystroem]
        X_final = sparse.hstack([X_original, sparse.csr_matrix(X_transformed)]).tocsr()
        return X_final
    
    # 3. Fallback: If transformer is missing but model expects it
    else:
        # We return X_original, but this will likely cause a crash in prediction
        # which we handle in the main loop
        return X_original

# ------------------------------
# Flask Routes
# ------------------------------
@app.route('/')
def index():
    # Pass None so the HTML doesn't crash looking for these variables
    return render_template('index.html', predictions=None, image_url=None)

@app.route("/analyse", methods=["POST"])
def analyse():
    # 1. Check if image exists
    if "image" not in request.files:
        return "⚠️ No image uploaded!", 400

    image_file = request.files["image"]
    if image_file.filename == "":
        return "⚠️ No file selected!", 400

    # 2. Save uploaded image
    filename = image_file.filename
    save_path = os.path.join(app.config["IMAGE_UPLOADS"], filename)
    image_file.save(save_path)
    print(f"📂 Image saved to: {save_path}")

    # 3. Process Image
    X_original = process_image(save_path)
    
    # 4. Initialize predictions
    labels = ["humour", "sarcastic", "offensive", "motivational", "overall"]
    predictions = {}

    # 5. Prediction Loop
    for label in labels:
        try:
            model_path = f"models/{label}_model.pkl"
            model = pickle.load(open(model_path, "rb"))
            
            # Prepare Features
            X_fit = prepare_for_prediction(X_original, label, NYSTROEM_TRANSFORMER)
            
            # Predict
            raw_pred = model.predict(X_fit)[0]
            readable_label = label_mapping.get(label, {}).get(raw_pred, raw_pred)
            predictions[label] = readable_label
            
        except Exception as e:
            print(f"[ERROR] {label}: {e}")
            predictions[label] = "Error"

    # 6. Generate Correct Image URL for the HTML
    # This uses Flask's internal router to find the file in 'static/uploads/'
    final_image_url = url_for('static', filename='uploads/' + filename)

    return render_template('index.html', predictions=predictions, image_url=final_image_url)
# ------------------------------
# Run App
# ------------------------------
if __name__ == "__main__":
    app.run(debug=True)