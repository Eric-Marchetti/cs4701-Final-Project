import pandas as pd
from gensim.models import Word2Vec
import numpy as np
import logistic_regression
import embedding


if __name__ == "__main__":
    df = pd.read_csv('data.csv')
    corpus = embedding.build_corpus(df)
    
    w2vmodel = Word2Vec(corpus, vector_size=100, window=5, min_count=1, workers=4)
    embeddings = embedding.get_tweet_embedding(w2vmodel,corpus)
    embeddings = embeddings.reshape(embeddings.shape[0], -1)
    labels = df['label'].values

    logistic_regression_model = logistic_regression.LogisticRegression()
    logistic_regression_model.train(embeddings, labels)

    preds = logistic_regression_model.predict(embeddings)
    loss = logistic_regression_model.loss(preds, labels)
    print(loss)