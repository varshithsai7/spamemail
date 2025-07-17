# Spam Email/SMS Classifier Project Guide

# This script will guide you through building a simple yet effective Spam Email/SMS Classifier
# using Python, scikit-learn, and common NLP techniques.

# Project Goal: To classify incoming text messages (or emails) as either "spam" or "ham" (legitimate).

# --- Step 1: Setup and Data Acquisition ---
# First, ensure you have the necessary libraries installed.
# If not, run:
# pip install pandas scikit-learn nltk

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re # Regular expression library

# Download NLTK data (run this once if you haven't already)
nltk.download('stopwords')
nltk.download('punkt') 
# # For tokenization

# Load the dataset
# You can find a common dataset for SMS Spam Collection on Kaggle (e.g., 'spam.csv').
# Make sure the CSV file is in the same directory as your script, or provide the full path.
try:
    df = pd.read_csv('spam.csv', encoding='latin-1')
    # The dataset often has unnamed columns, let's clean them up and rename relevant ones.
    df = df.iloc[:, :2] # Keep only the first two columns
    df.columns = ['label', 'message'] # Rename columns for clarity
    print("Dataset loaded successfully!")
    print(df.head())
    print("\nDataset Info:")
    df.info()
    print("\nLabel Distribution:")
    print(df['label'].value_counts())
except FileNotFoundError:
    print("Error: 'spam.csv' not found. Please download it from Kaggle or provide the correct path.")
    print("You can download it from: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset")
    exit() # Exit if the file isn't found

# --- Step 2: Data Preprocessing ---
# This is crucial for text data. We'll clean the text messages.

# Initialize stemmer and stopwords
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # 1. Convert to lowercase
    text = text.lower()
    # 2. Remove punctuation and numbers
    text = re.sub(r'[^a-z\s]', '', text) # Keep only letters and spaces
    # 3. Tokenization (split text into words)
    words = nltk.word_tokenize(text)
    # 4. Remove stopwords and apply stemming
    processed_words = [stemmer.stem(word) for word in words if word not in stop_words]
    # 5. Join words back into a single string
    return ' '.join(processed_words)

# Apply preprocessing to all messages
df['processed_message'] = df['message'].apply(preprocess_text)
print("\nProcessed Messages Sample:")
print(df[['message', 'processed_message']].head())

# --- Step 3: Feature Extraction (Text Vectorization) ---
# Convert text data into numerical features that ML models can understand.
# TF-IDF (Term Frequency-Inverse Document Frequency) is a common and effective method.

# Initialize TF-IDF Vectorizer
# max_features limits the number of features (words) to consider, focusing on most frequent ones.
tfidf_vectorizer = TfidfVectorizer(max_features=5000)

# Fit and transform the processed messages
X = tfidf_vectorizer.fit_transform(df['processed_message'])
y = df['label'].map({'ham': 0, 'spam': 1}) # Convert labels to numerical (0 for ham, 1 for spam)

print(f"\nShape of TF-IDF features (X): {X.shape}")
print(f"Shape of labels (y): {y.shape}")

# --- Step 4: Split Data and Model Training ---
# Split data into training and testing sets to evaluate the model's performance on unseen data.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

# Initialize and train a Multinomial Naive Bayes classifier
# Naive Bayes is a simple yet powerful algorithm for text classification.
model = MultinomialNB()
model.fit(X_train, y_train)

print("\nModel training complete!")

# --- Step 5: Model Evaluation ---
# Evaluate the model's performance on the test set.

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\n--- Model Evaluation ---")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}") # Precision: Out of all predicted spam, how many were actually spam? (Minimizes false positives)
print(f"Recall:    {recall:.4f}")    # Recall: Out of all actual spam, how many did we correctly identify? (Minimizes false negatives)
print(f"F1-Score:  {f1:.4f}")      # F1-Score: Harmonic mean of Precision and Recall (balances both)

# --- Step 6: Making Predictions on New Messages ---
# How to use your trained model for new, unseen messages.

def predict_spam(message):
    # Preprocess the new message
    processed_message = preprocess_text(message)
    # Vectorize the processed message using the SAME TF-IDF vectorizer fitted on training data
    vectorized_message = tfidf_vectorizer.transform([processed_message])
    # Make prediction
    prediction = model.predict(vectorized_message)
    # Convert numerical prediction back to label
    return "SPAM" if prediction[0] == 1 else "HAM"

# Test with some example messages
print("\n--- Testing with New Messages ---")
test_message_1 = "Congratulations! You've won a free iPhone. Click here to claim."
print(f"'{test_message_1}' is: {predict_spam(test_message_1)}")

test_message_2 = "Hey, let's meet for coffee tomorrow at 10 AM."
print(f"'{test_message_2}' is: {predict_spam(test_message_2)}")

test_message_3 = "URGENT! Your bank account has been compromised. Verify your details now."
print(f"'{test_message_3}' is: {predict_spam(test_message_3)}")

test_message_4 = "Hi, how are you doing today? Just checking in."
print(f"'{test_message_4}' is: {predict_spam(test_message_4)}")

# --- Further Improvements / Next Steps ---
# 1. Try other classifiers: Logistic Regression, Support Vector Machines (SVM), Random Forest.
# 2. Experiment with different text vectorization techniques (e.g., CountVectorizer, Word Embeddings like Word2Vec if you want to delve into deep learning).
# 3. Handle imbalanced datasets more explicitly (e.g., using SMOTE for oversampling).
# 4. Build a simple web interface (using Flask or Streamlit) to interact with your model.
# 5. Deploy your model to a cloud platform (e.g., Heroku, AWS Lambda, Google Cloud Functions).
