import pandas as pd
from sklearn.model_selection import train_test_split

# Read the data
df = pd.read_csv('labelled_tweets.csv', 
                 quoting=1,
                 escapechar='\\',
                 index_col='id',
                 dtype={
                     'id': int,
                     'created_at': str,
                     'full_text': str,
                     'score': float
                 })

df['created_at'] = pd.to_datetime(df['created_at'])

total_len = len(df)
train_size = int(total_len * 0.7)
val_size = int(total_len * 0.15)
test_size = total_len - train_size - val_size

train = df.iloc[:train_size]
val = df.iloc[train_size:(train_size + val_size)]
test = df.iloc[(train_size + val_size):]
# Save to CSV files
train.to_csv('train.csv')
val.to_csv('val.csv')
test.to_csv('test.csv')

# Print the sizes of the splits
print(f"Total: {total_len}, Train: {train_size}, Validation: {val_size}, Test: {test_size}")
