# spamemail

## Project Overview

This project is a Spam Email/SMS Classifier built using Python, scikit-learn, and natural language processing (NLP) techniques. Its primary goal is to automatically classify incoming text messages or emails as either "spam" (unwanted or malicious) or "ham" (legitimate).

## Features
- **Machine Learning Models:** Utilizes models such as Logistic Regression, Multinomial Naive Bayes, and Support Vector Classifier for robust spam detection.
- **Text Preprocessing:** Cleans and processes text using tokenization, stemming, and stopword removal to improve classification accuracy.
- **TF-IDF Vectorization:** Converts text data into numerical features using TF-IDF for effective model training.
- **Streamlit Web App:** Includes an interactive web interface where users can input messages and instantly see if they are spam or not.
- **Custom Prediction:** Allows users to test the classifier with their own messages.

## Usage
1. **Dataset:** The project uses the popular SMS Spam Collection dataset (`spam.csv`). Download it from [Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) and place it in the project directory.
2. **Run the App:**
   - Install dependencies: `pip install -r requirements.txt`
   - Start the Streamlit app: `streamlit run app.py`
3. **Try It Out:** Enter any message in the web app to check if it's spam or ham.

## How It Works
- The app preprocesses the input message, vectorizes it, and uses a trained machine learning model to predict whether the message is spam or not.
- The model is trained on a labeled dataset and can be easily retrained or extended with new data.

## Applications
- Email and SMS spam filtering
- Automated moderation for messaging platforms
- Educational tool for learning about NLP and machine learning

---
Developed on 17 July.