import os
import json
import pandas as pd
import numpy as np
from pathlib import Path

import gzip

# Config
# Link to the dataset: https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Video_Games_5.json.gz
INPUT_FILE = "reviews_Video_Games_5.json.gz"
OUTPUT_DIR = Path("datasets/video_games")
# Since the file is ALREADY 5-core filtered from the website, we can just leave this as a safety check
MIN_INTERACTIONS = 5
VAL_USERS_COUNT = 1024 # Number of users to use for validation

def main():
    print(f"Reading {INPUT_FILE}...")
    
    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: File not found at {INPUT_FILE}. Please update the INPUT_FILE path in the script.")
        return

    records = []
    with gzip.open(INPUT_FILE, 'rt', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            # Handle different Amazon format keys (2014, 2018, 2023 versions)
            user = data.get('reviewerID', data.get('user_id'))
            item = data.get('asin', data.get('parent_asin', data.get('item_id')))
            time = data.get('unixReviewTime', data.get('timestamp'))
            
            if user and item and time:
                records.append({'user': user, 'item': item, 'time': float(time)})
                
    df = pd.DataFrame(records)
    print(f"Original dataset: {len(df)} interactions, {df['user'].nunique()} users, {df['item'].nunique()} items.")
    
    # 5-core iterative filtering
    print(f"Applying {MIN_INTERACTIONS}-core filtering (removing noise)...")
    while True:
        user_counts = df['user'].value_counts()
        item_counts = df['item'].value_counts()
        
        valid_users = user_counts[user_counts >= MIN_INTERACTIONS].index
        valid_items = item_counts[item_counts >= MIN_INTERACTIONS].index
        
        df_filtered = df[df['user'].isin(valid_users) & df['item'].isin(valid_items)]
        
        if len(df_filtered) == len(df):
            break
        df = df_filtered
        
    print(f"Filtered dataset: {len(df)} interactions, {df['user'].nunique()} users, {df['item'].nunique()} items.")
    
    # Sort chronologically
    print("Sorting interactions chronologically...")
    df = df.sort_values(by=['user', 'time'])
    
    # Map IDs to 1..N
    user_map = {u: i+1 for i, u in enumerate(df['user'].unique())}
    item_map = {i: j+1 for j, i in enumerate(df['item'].unique())}
    
    df['user_id'] = df['user'].map(user_map)
    df['item_id'] = df['item'].map(item_map)
    
    # Group into sequences
    user_items = df.groupby('user_id')['item_id'].apply(list).to_dict()
    
    # Create directories
    (OUTPUT_DIR / "train").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "val").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "test").mkdir(parents=True, exist_ok=True)
    
    # Dataset stats
    dataset_stats = {
        "num_users": len(user_map),
        "num_items": len(item_map), 
        "num_interactions": len(df)
    }
    
    with open(OUTPUT_DIR / "dataset_stats.json", "w") as f:
        json.dump(dataset_stats, f, indent=4)
        
    print("Splitting into Train/Val/Test...")
    rng = np.random.RandomState(42)
    val_users = set(rng.choice(list(user_map.values()), min(VAL_USERS_COUNT, len(user_map)), replace=False))
    
    train_sequences = []
    val_input_sequences, val_gt_actions = [], []
    test_input_sequences, test_gt_actions = [], []
    
    for user_id, seq in user_items.items():
        if len(seq) < 3:
            continue
            
        if user_id in val_users:
            train_sequences.append(seq[:-3])
            
            val_input_sequences.append(seq[:-2])
            val_gt_actions.append(seq[-2])
            
            test_input_sequences.append(seq[:-1])
            test_gt_actions.append(seq[-1])
        else:
            train_sequences.append(seq[:-2])
            
            test_input_sequences.append(seq[:-1])
            test_gt_actions.append(seq[-1])
            
    # Write to files
    print("Writing files...")
    with open(OUTPUT_DIR / "train" / "input.txt", "w") as f:
        for seq in train_sequences:
            if len(seq) > 0: f.write(" ".join(map(str, seq)) + "\n")
            
    with open(OUTPUT_DIR / "val" / "input.txt", "w") as f:
        for seq in val_input_sequences:
            f.write(" ".join(map(str, seq)) + "\n")
            
    with open(OUTPUT_DIR / "val" / "output.txt", "w") as f:
        for action in val_gt_actions:
            f.write(str(action) + "\n")
            
    with open(OUTPUT_DIR / "test" / "input.txt", "w") as f:
        for seq in test_input_sequences:
            f.write(" ".join(map(str, seq)) + "\n")
            
    with open(OUTPUT_DIR / "test" / "output.txt", "w") as f:
        for action in test_gt_actions:
            f.write(str(action) + "\n")
            
    print("Preprocessing complete!")

if __name__ == "__main__":
    main()
