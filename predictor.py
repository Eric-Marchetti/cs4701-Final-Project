import joblib
import torch
from transformers import BertTokenizer, BertForSequenceClassification, GPT2Tokenizer, GPT2ForSequenceClassification
import numpy as np
import os

from text_preprocessor import clean_tweet_text

LOGISTIC_MODEL_PATH = 'saved_models/logistic_regression_model.joblib'
VECTORIZER_PATH = 'saved_models/tfidf_vectorizer.joblib'
BERT_MODEL_DIR = 'saved_models/bert_sentiment_model'
GPT2_MODEL_DIR = 'saved_models/gpt2_sentiment_model'

if os.path.exists(LOGISTIC_MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
    lr_model = joblib.load(LOGISTIC_MODEL_PATH)
    tfidf_vectorizer = joblib.load(VECTORIZER_PATH)
else:
    lr_model = None
    tfidf_vectorizer = None
    print("Warning: Logistic Regression model or vectorizer not found. Run logistic_regression_model.py first.")

if os.path.exists(BERT_MODEL_DIR):
    bert_tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_DIR)
    bert_model = BertForSequenceClassification.from_pretrained(BERT_MODEL_DIR)
    bert_model.eval()
else:
    bert_tokenizer = None
    bert_model = None
    print("Warning: BERT model not found. Run bert_model.py first.")

if os.path.exists(GPT2_MODEL_DIR):
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained(GPT2_MODEL_DIR)
    gpt2_model = GPT2ForSequenceClassification.from_pretrained(GPT2_MODEL_DIR)
    if gpt2_tokenizer.pad_token is None:
        gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
        gpt2_model.config.pad_token_id = gpt2_tokenizer.pad_token_id
    gpt2_model.eval()
else:
    gpt2_tokenizer = None
    gpt2_model = None
    print("Warning: GPT-2 model not found. Run gpt2_model.py first.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if bert_model: bert_model.to(device)
if gpt2_model: gpt2_model.to(device)

def predict_logistic_regression(texts):
    if not lr_model or not tfidf_vectorizer:
        print("Logistic Regression model/vectorizer not loaded.")
        return [None] * len(texts)
    cleaned_texts = [clean_tweet_text(text) for text in texts]
    if not any(cleaned_texts):
        return [0] * len(texts)
    
    non_empty_texts = [t for t in cleaned_texts if t]
    if not non_empty_texts:
         return [0] * len(texts)

    text_features = tfidf_vectorizer.transform(non_empty_texts)
    predictions = lr_model.predict(text_features)

    final_predictions = []
    pred_idx = 0
    for text in cleaned_texts:
        if text:
            final_predictions.append(predictions[pred_idx])
            pred_idx += 1
        else:
            final_predictions.append(0)
    return final_predictions

def predict_transformer_model(texts, model, tokenizer, model_name="Transformer", batch_size=32):
    if not model or not tokenizer:
        print(f"{model_name} model/tokenizer not loaded.")
        return [None] * len(texts)

    all_predictions = []
    model.eval()

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        if not batch_texts:
            continue

        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1).cpu().numpy()
            all_predictions.extend(predictions.tolist())
        
        if (i // batch_size) % 100 == 0:
            print(f"Processed {i + len(batch_texts)}/{len(texts)} for {model_name}")

    return all_predictions

def predict_bert(texts):
    return predict_transformer_model(texts, bert_model, bert_tokenizer, "BERT")

def predict_gpt2(texts):
    return predict_transformer_model(texts, gpt2_model, gpt2_tokenizer, "GPT-2")