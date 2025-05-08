import pandas as pd
from gensim.models import Word2Vec
import numpy as np
import logistic_regression
import embedding
from sklearn.metrics import accuracy_score, classification_report
import pickle

def load_and_process(file_path):
    df = pd.read_csv(file_path)
    corpus = embedding.build_corpus(df.rename(columns={"full_text": "text"}))
    return corpus, df['score'].apply(lambda x: 1 if x > 0 else 0).values  # Binarize score to positive/negative

if __name__ == "__main__":
    train_corpus, y_train = load_and_process('train.csv')
    val_corpus, y_val = load_and_process('val.csv')
    test_corpus, y_test = load_and_process('test.csv')

    #get the embeddings
    w2vmodel = Word2Vec(train_corpus, vector_size=100, window=5, min_count=1, workers=4)
    
    
    X_train = embedding.get_tweet_embedding(w2vmodel,[" ".join(tweet) for tweet in train_corpus])
    X_train = np.array(X_train).reshape(len(X_train), -1)

    X_val = embedding.get_tweet_embedding(w2vmodel, [" ".join(tweet) for tweet in val_corpus])
    X_val = np.array(X_val).reshape(len(X_val), -1)

    X_test = embedding.get_tweet_embedding(w2vmodel, [" ".join(tweet) for tweet in test_corpus])
    X_test = np.array(X_test).reshape(len(X_test), -1)

    # Train model
    model = logistic_regression.LogisticRegression(learning_rate=0.01, num_iterations=1000)
    model.train(X_train, y_train)

    # Evaluate
    with open("model_results.txt", "w") as f:
        for split, X, y in zip(["Validation", "Test"], [X_val, X_test], [y_val, y_test]):
            preds = model.predict(X)
            acc = accuracy_score(y, preds)
            report = classification_report(y, preds)
            print(f"{split} Accuracy: {acc:.4f}")
            print(f"{split} Classification Report:\n", report)
            f.write(f"{split} Accuracy: {acc:.4f}\n")
            f.write(f"{split} Classification Report:\n{report}\n")
            f.write("-" * 50 + "\n")

    
    with open("logistic_model.pkl", "wb") as f:
        pickle.dump(model, f)

    w2vmodel.save("word2vec.model")