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
from sklearn.linear_model import LogisticRegression # New import for Logistic Regression
from sklearn.svm import SVC # New import for Support Vector Classifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re # Regular expression library

# Download NLTK data (run this once if you haven't already)
# nltk.download('stopwords')
# nltk.download('punkt') # For tokenization
# nltk.download('punkt_tab') # Ensure this is also downloaded for word_tokenize

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

# --- Step 4: Split Data ---
# Split data into training and testing sets to evaluate the model's performance on unseen data.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

# --- Step 5: Model Training and Evaluation (Multinomial Naive Bayes) ---
print("\n--- Training and Evaluating Multinomial Naive Bayes ---")
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
nb_y_pred = nb_model.predict(X_test)

nb_accuracy = accuracy_score(y_test, nb_y_pred)
nb_precision = precision_score(y_test, nb_y_pred)
nb_recall = recall_score(y_test, nb_y_pred)
nb_f1 = f1_score(y_test, nb_y_pred)

print(f"Multinomial Naive Bayes Accuracy:  {nb_accuracy:.4f}")
print(f"Multinomial Naive Bayes Precision: {nb_precision:.4f}")
print(f"Multinomial Naive Bayes Recall:    {nb_recall:.4f}")
print(f"Multinomial Naive Bayes F1-Score:  {nb_f1:.4f}")


# --- Step 5.1: Model Training and Evaluation (Logistic Regression) ---
print("\n--- Training and Evaluating Logistic Regression ---")
# Logistic Regression is a good baseline classifier and often performs well.
# max_iter is increased to ensure convergence for larger datasets.
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)
lr_y_pred = lr_model.predict(X_test)

lr_accuracy = accuracy_score(y_test, lr_y_pred)
lr_precision = precision_score(y_test, lr_y_pred)
lr_recall = recall_score(y_test, lr_y_pred)
lr_f1 = f1_score(y_test, lr_y_pred)

print(f"Logistic Regression Accuracy:  {lr_accuracy:.4f}")
print(f"Logistic Regression Precision: {lr_precision:.4f}")
print(f"Logistic Regression Recall:    {lr_recall:.4f}")
print(f"Logistic Regression F1-Score:  {lr_f1:.4f}")


# --- Step 5.2: Model Training and Evaluation (Support Vector Classifier - SVC) ---
print("\n--- Training and Evaluating Support Vector Classifier (SVC) ---")
# SVC can be powerful but might take longer to train on larger datasets.
svc_model = SVC(kernel='linear', random_state=42) # 'linear' kernel is often good for text data
svc_model.fit(X_train, y_train)
svc_y_pred = svc_model.predict(X_test)

svc_accuracy = accuracy_score(y_test, svc_y_pred)
svc_precision = precision_score(y_test, svc_y_pred)
svc_recall = recall_score(y_test, svc_y_pred)
svc_f1 = f1_score(y_test, svc_y_pred)

print(f"SVC Accuracy:  {svc_accuracy:.4f}")
print(f"SVC Precision: {svc_precision:.4f}")
print(f"SVC Recall:    {svc_recall:.4f}")
print(f"SVC F1-Score:  {svc_f1:.4f}")


# --- Step 6: Making Predictions on New Messages (using the best performing model, or a chosen one) ---
# For demonstration, we'll use the Logistic Regression model for predictions here,
# as it often offers a good balance of performance and speed.
# You can switch this to nb_model or svc_model if their metrics are better for your goal.

def predict_spam(message, chosen_model, vectorizer):
    # Preprocess the new message
    processed_message = preprocess_text(message)
    # Vectorize the processed message using the SAME TF-IDF vectorizer fitted on training data
    vectorized_message = vectorizer.transform([processed_message])
    # Make prediction
    prediction = chosen_model.predict(vectorized_message)
    # Convert numerical prediction back to label
    return "SPAM" if prediction[0] == 1 else "HAM"

# Test with some example messages using the Logistic Regression model
print("\n--- Testing with New Messages (using Logistic Regression Model) ---")
test_message_1 = "Congratulations! You've won a free iPhone. Click here to claim."
print(f"'{test_message_1}' is: {predict_spam(test_message_1, lr_model, tfidf_vectorizer)}")

test_message_2 = "Hey, let's meet for coffee tomorrow at 10 AM."
print(f"'{test_message_2}' is: {predict_spam(test_message_2, lr_model, tfidf_vectorizer)}")

test_message_3 = "URGENT! Your bank account has been compromised. Verify your details now."
print(f"'{test_message_3}' is: {predict_spam(test_message_3, lr_model, tfidf_vectorizer)}")

test_message_4 = "Hi, how are you doing today? Just checking in."
print(f"'{test_message_4}' is: {predict_spam(test_message_4, lr_model, tfidf_vectorizer)}")

# --- Further Improvements / Next Steps ---
# 1. Compare the performance metrics of all three models (Naive Bayes, Logistic Regression, SVC).
#    Which one is "best" depends on your priority (e.g., maximizing precision to avoid false positives,
#    or maximizing recall to catch all spam). For spam, high precision is usually preferred.
# 2. Experiment with different text vectorization techniques (e.g., CountVectorizer, Word Embeddings like Word2Vec if you want to delve into deep learning).
# 3. Handle imbalanced datasets more explicitly (e.g., using SMOTE for oversampling).
# 4. Build a simple web interface (using Flask or Streamlit) to interact with your model.
# 5. Deploy your model to a cloud platform (e.g., Heroku, AWS Lambda, Google Cloud Functions).
