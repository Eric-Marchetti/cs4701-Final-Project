import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os

from data_loader import load_labeled_tweets
from text_preprocessor import clean_tweet_text

MODEL_NAME = "bert-base-uncased" 
BERT_MODEL_DIR = 'saved_models/bert_sentiment_model'

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def train_bert_model():
    print("Loading labeled data...")
    labeled_df = load_labeled_tweets()

    print("Preprocessing text data (optional)...")
    labeled_df['text'] = labeled_df['text'].apply(lambda x: clean_tweet_text(x) if isinstance(x, str) else "")
    labeled_df = labeled_df[labeled_df['text'] != '']
    if labeled_df.empty:
        print("No data remaining after cleaning for BERT. Exiting.")
        return
    # labels: 0 (neutral), 1 (positive), 2 (negative)
    print(f"Label distribution:\n{labeled_df['label'].value_counts()}")

    train_df, eval_df = train_test_split(labeled_df, test_size=0.2, random_state=42, stratify=labeled_df['label'])

    train_dataset = Dataset.from_pandas(train_df[['text', 'label']].reset_index(drop=True))
    eval_dataset = Dataset.from_pandas(eval_df[['text', 'label']].reset_index(drop=True))
    
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': eval_dataset
    })

    print("Loading BERT tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_function(examples):
        return tokenizer([str(text) for text in examples['text']], padding="max_length", truncation=True, max_length=128)

    print("Tokenizing datasets...")
    tokenized_datasets = dataset_dict.map(tokenize_function, batched=True)

    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets.set_format("torch")

    print("Loading BERT model for sequence classification...")
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3) 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    training_args = TrainingArguments(
        output_dir='./results/bert',          
        num_train_epochs=1,
        per_device_train_batch_size=16,  
        per_device_eval_batch_size=16,   
        warmup_steps=500,                
        weight_decay=0.01,               
        logging_dir='./logs/bert',            
        logging_steps=100,
        save_strategy="no",
        load_best_model_at_end=True,
        report_to="none"
    )

    trainer = Trainer(
        model=model,                         
        args=training_args,                  
        train_dataset=tokenized_datasets["train"],       
        eval_dataset=tokenized_datasets["validation"],
        compute_metrics=compute_metrics,
        tokenizer=tokenizer
    )

    print("Starting BERT model training...")
    trainer.train()

    print("Evaluating BERT model...")
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")

    print("Saving fine-tuned BERT model and tokenizer...")
    os.makedirs(BERT_MODEL_DIR, exist_ok=True)
    model.save_pretrained(BERT_MODEL_DIR)
    tokenizer.save_pretrained(BERT_MODEL_DIR)
    print(f"BERT model and tokenizer saved to {BERT_MODEL_DIR}")

if __name__ == '__main__':
    train_bert_model() 