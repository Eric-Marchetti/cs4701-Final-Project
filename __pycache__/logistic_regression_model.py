import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

from data_loader import load_labeled_tweets
from text_preprocessor import clean_tweet_text

MODEL_DIR = 'saved_models'
LOGISTIC_MODEL_PATH = os.path.join(MODEL_DIR, 'logistic_regression_model.joblib')
VECTORIZER_PATH = os.path.join(MODEL_DIR, 'tfidf_vectorizer.joblib')

def train_logistic_regression():
    print("Loading labeled data...")
    labeled_df = load_labeled_tweets()

    print("Preprocessing text data...")
    labeled_df['cleaned_text'] = labeled_df['text'].apply(clean_tweet_text)

    labeled_df = labeled_df[labeled_df['cleaned_text'] != '']
    
    if labeled_df.empty:
        print("No data remaining after cleaning. Exiting.")
        return

    X = labeled_df['cleaned_text']
    y = labeled_df['label'] # Labels are 0: neutral, 1: positive, 2: negative

    print(f"Data shapes: X - {X.shape}, y - {y.shape}")
    print(f"Class distribution:\n{y.value_counts(normalize=True)}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Vectorizing text data using TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print("Training Logistic Regression model...")
    model = LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced', max_iter=1000)
    model.fit(X_train_tfidf, y_train)

    print("Evaluating model...")
    y_pred = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Neutral (0)', 'Positive (1)', 'Negative (2)'])

    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:\n", report)

    print("Saving model and vectorizer...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, LOGISTIC_MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"Model saved to {LOGISTIC_MODEL_PATH}")
    print(f"Vectorizer saved to {VECTORIZER_PATH}")

if __name__ == '__main__':
    train_logistic_regression() 