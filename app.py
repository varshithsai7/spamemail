import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import os

# --- NLTK Data Downloads (ensure these are done, ideally once outside the app) ---
try:
    nltk.data.find('corpora/stopwords.zip')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt.zip')
except nltk.downloader.DownloadError:
    nltk.download('punkt')
try:
    nltk.data.find('tokenizers/punkt_tab.zip')
except nltk.downloader.DownloadError:
    nltk.download('punkt_tab')

# --- Step 1: Data Preprocessing Function ---
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = nltk.word_tokenize(text)
    processed_words = [stemmer.stem(word) for word in words if word not in stop_words]
    return ' '.join(processed_words)

# --- Step 2: Model Loading and Training ---
@st.cache_resource # Cache the model and vectorizer to avoid retraining on every interaction
def train_model():
    # Check if spam.csv exists
    if not os.path.exists('spam.csv'):
        st.error("Error: 'spam.csv' not found. Please download it from Kaggle and place it in the same directory as this script.")
        st.stop() # Stop the app execution if file is missing

    df = pd.read_csv('spam.csv', encoding='latin-1')
    df = df.iloc[:, :2]
    df.columns = ['label', 'message']
    df['processed_message'] = df['message'].apply(preprocess_text)

    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    X = tfidf_vectorizer.fit_transform(df['processed_message'])
    y = df['label'].map({'ham': 0, 'spam': 1})

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    return model, tfidf_vectorizer

# Load the model and vectorizer
model, tfidf_vectorizer = train_model()

# --- Step 3: Prediction Function ---
def predict_spam(message, trained_model, vectorizer):
    processed_message = preprocess_text(message)
    vectorized_message = vectorizer.transform([processed_message])
    prediction = trained_model.predict(vectorized_message)
    return "SPAM" if prediction[0] == 1 else "HAM"

# --- Step 4: Streamlit App Interface ---
st.set_page_config(page_title="Spam Classifier", layout="centered") # Revert to centered layout

st.markdown(
    """
    <style>
    /* Ensure all text within the main Streamlit app is readable */
    body {
        color: #333; /* Default dark text color */
    }
    .main {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        color: #333; /* Ensure main content text is dark */
    }
    .stTextInput>div>div>textarea {
        border-radius: 10px;
        border: 1px solid #ccc;
        padding: 10px;
        color: #333; /* Ensure input text is dark */
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        font-size: 16px;
        border: none;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stAlert {
        border-radius: 10px;
    }
    .header-text {
        color: #333;
        font-family: 'Inter', sans-serif;
        text-align: center;
        margin-bottom: 20px;
    }
    .sub-header-text {
        color: #555;
        font-family: 'Inter', sans-serif;
        text-align: center;
        margin-bottom: 30px;
    }
    .result-text {
        font-size: 1.5em;
        font-weight: bold;
        margin-top: 20px; /* Add some space above the result */
    }
    .spam-color {
        color: #dc3545; /* Red for SPAM */
    }
    .ham-color {
        color: #28a745; /* Green for HAM */
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 class='header-text'>📧 Spam or Ham Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header-text'>Enter any message below to check if it's SPAM or HAM!</p>", unsafe_allow_html=True)

# Text input for the message
user_message = st.text_area("Enter your message here:", height=150, placeholder="Type your message...")

# Prediction button
if st.button("Analyze Message"):
    if user_message:
        # Get prediction
        prediction = predict_spam(user_message, model, tfidf_vectorizer)

        # Display result using st.markdown for HTML formatting
        if prediction == "SPAM":
            st.markdown(f"<p class='result-text spam-color'>Prediction: This message is likely <strong>{prediction}</strong> 🚫</p>", unsafe_allow_html=True)
        else:
            st.markdown(f"<p class='result-text ham-color'>Prediction: This message is <strong>{prediction}</strong> ✅</p>", unsafe_allow_html=True)
    else:
        st.warning("Please enter a message to analyze.")

st.markdown("---")
st.markdown("Developed on 17 july.")