import sys
import pandas as pd
sys.path.append(".")

from src.data.loader import load_config, load_yelp_reviews, load_amazon_reviews
from src.data.preprocessor import preprocess, split_data, save_splits

config = load_config()

# Load small sample from both
print("Loading data...")
yelp_df = load_yelp_reviews(
    filepath=config["data"]["yelp"]["review_file"],
    max_rows=5000
)
amazon_df = load_amazon_reviews(
    filepath="data/raw/All_Beauty.jsonl",
    max_rows=5000
)

# Combine
combined = pd.concat([yelp_df, amazon_df], ignore_index=True)
print(f"Combined: {len(combined):,} reviews")

# Preprocess
cleaned = preprocess(combined, config)
print(f"After cleaning: {len(cleaned):,} reviews")

# Split
train_df, val_df, test_df = split_data(cleaned, config)

# Save
save_splits(train_df, val_df, test_df, config)

# Verify saved files
import os
for split in ["train", "val", "test"]:
    path = f"data/processed/{split}.csv"
    df = pd.read_csv(path)
    print(f"{split}: {len(df):,} rows saved at {path}")

print("\nPreprocessor test passed.")