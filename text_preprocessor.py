import re

def clean_tweet_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()  # Lowercase
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'\@\w+', '', text)  # Remove mentions
    text = re.sub(r'#', '', text) # Remove hashtag symbol but keep the text
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters and punctuation
    text = re.sub(r'\s+', ' ', text).strip() # Remove extra spaces
    return text