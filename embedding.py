import numpy as np
import re


def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+|@\w+|#\w+|[^a-z\s]", "", text)  # Remove URLs, mentions, hashtags, punctuation
    return text.strip().split()

def build_corpus(data):
    return data['text'].apply(preprocess).tolist()

def get_word_embeddings(model, sequence, max_len=100):
    word_embeddings = []
    if isinstance(sequence, str): #not sure if itll be a list or one long string
      sequence = sequence.split()
    for word in sequence:
        if word in model.wv.key_to_index: # check if words in vocab
            word_embeddings.append(model.wv[word])
        else:
            word_embeddings.append(np.zeros(model.vector_size)) 
            #gotta keep all embeddings the same size for LR later
    
    while len(word_embeddings) < max_len:  # Pad
        word_embeddings.append(np.zeros(model.vector_size))
    return np.array(word_embeddings)


def get_tweet_embedding(model, data):
    tweet_embedding = []
    for tweet in data:
        word_embeddings = get_word_embeddings(model, tweet.split())
        tweet_embedding.append(word_embeddings)
    return tweet_embedding


