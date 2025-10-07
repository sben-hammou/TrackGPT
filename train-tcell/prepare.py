import os
import tiktoken
import numpy as np
import pandas as pd

df = pd.read_csv('train-tcell\tcell_tracks1.csv') # read dataframe
clean = df.drop(columns=['track_no', 'tframe']) # exclude track_no column and tframe column

## subset = clean.head(500) # take first 500 rows(for faster training)
data = clean.groupby(df['track_no']).apply(
    lambda g: f"B\n{g.to_string(index=False, header=False)}\nE"  # add "B" and "E" tokens around each track
).str.cat(sep='\n\n')
data += '\n\n' 

n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode with tiktoken gpt2 bpe
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# train.bin has 301,966 tokens
# val.bin has 36,059 tokens
