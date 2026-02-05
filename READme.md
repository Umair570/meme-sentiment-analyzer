# Multi-Modal Meme Classification AI 🤖🖼️

A full-stack AI application that analyzes memes using **Computer Vision** and **Natural Language Processing (NLP)**. This system extracts text from images using OCR, analyzes visual features using ResNet50, and classifies memes across multiple psychological dimensions using advanced machine learning techniques.

## 🚀 Features

* **Multi-Modal Analysis:** Combines text (OCR) and image features (ResNet50) for higher accuracy.
* **Advanced Feature Engineering:** Utilizes **Nystroem Kernel Approximation** to map linear features into non-linear space for SGD classifiers.
* **5-Point Classification System:** Automatically tags memes into five distinct categories:
    * **Humour:** (Funny / Not Funny / Sarcastic)
    * **Sarcasm:** (Sarcastic / Not Sarcastic)
    * **Offense:** (Offensive / Not Offensive)
    * **Motivation:** (Motivational / Not Motivational)
    * **Overall Sentiment:** (Positive / Negative)
* **Web Interface:** Built with **Flask** to allow users to upload and analyze images in real-time.

## 🛠️ Tech Stack

* **Language:** Python 3.x
* **Web Framework:** Flask
* **Deep Learning:** TensorFlow / Keras (ResNet50)
* **Machine Learning:** Scikit-Learn (SGD, Random Forest, TF-IDF, Nystroem)
* **OCR:** Tesseract (pytesseract)
* **Data Processing:** Pandas, NumPy, SciPy

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Install Teserract OCR**

Note: If you installed Tesseract in a custom location, update the path in app.py:
   ```bash
   #pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
   ```
Linux: sudo apt-get install tesseract-ocr
macOS: brew install tesseract   

3. **Train the models**

Before running the app, you must generate the machine learning models.
Place your dataset (train_cleaned_final.csv) in the root directory.
Run the training script:

   ```bash
   python train.py
   ```
This will generate vectorizers and model files inside the models/ directory.

4. **Run the Application**

    ```bash
    python app.py
    ```

📊 How It Works (The Science)
Image Input: The user uploads a meme.

OCR Extraction: pytesseract reads the text overlay.

Visual Processing: ResNet50 (pretrained on ImageNet) extracts 2048 deep visual features.

Text Vectorization: TF-IDF converts text into statistical vectors (unigrams + bigrams).

Fusion: Text and Image features are combined into a sparse matrix.

Transformation: A Nystroem Transformer approximates a kernel map to allow the linear SGD model to solve non-linear problems.

Prediction: Five separate models predict the attributes of the meme.

👨‍💻 Author
Muhammad Umair Ashraf

Note: This project was developed for educational and research purposes.