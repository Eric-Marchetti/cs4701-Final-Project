from datasets import load_dataset
import pandas as pd

def load_labeled_tweets():
    """
    Loads the labeled financial tweets dataset.
    Sentiment: 1 for bullish, 2 for bearish, and 0 for neutral.
    """
    dataset = load_dataset("TimKoornstra/financial-tweets-sentiment")
    df = dataset['train'].to_pandas()
    # Rename columns for clarity and consistency
    df = df.rename(columns={'tweet': 'text', 'sentiment': 'label'})
    # Map sentiment labels: 0 (neutral), 1 (positive/bullish), 2 (negative/bearish)
    # We'll keep these numerical labels for now.
    return df

def load_unlabeled_tweets():
    """
    Loads the unlabeled stock market tweets dataset.
    """
    dataset = load_dataset("StephanAkkerman/stock-market-tweets-data")
    df = dataset['train'].to_pandas()
    # Select relevant columns and rename if necessary
    df = df[['created_at', 'text']]
    return df

if __name__ == '__main__':
    print("Loading labeled dataset...")
    labeled_df = load_labeled_tweets()
    print(f"Labeled dataset shape: {labeled_df.shape}")
    print(labeled_df.head())
    print("\\nLabel distribution:")
    print(labeled_df['label'].value_counts())

    print("\\nLoading unlabeled dataset...")
    unlabeled_df = load_unlabeled_tweets()
    print(f"Unlabeled dataset shape: {unlabeled_df.shape}")
    print(unlabeled_df.head())
    print(f"Date range for unlabeled tweets: {unlabeled_df['created_at'].min()} to {unlabeled_df['created_at'].max()}") 