import pandas as pd

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
train_bound = int(total_len * 0.7)
train_val_bound = int(total_len * 0.85)

train = df.iloc[:train_bound]
val = df.iloc[train_bound:train_val_bound]
test = df.iloc[train_val_bound:]

train.to_csv('train.csv')
val.to_csv('val.csv')
test.to_csv('test.csv')

print(f"Total: {total_len}, Train: {train_bound}, Validation: {train_val_bound - train_bound}, Test: {total_len - train_val_bound}")
