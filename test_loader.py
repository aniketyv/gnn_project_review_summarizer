import sys
import pandas as pd
sys.path.append(".")

from src.data.loader import (
    load_config,
    load_yelp_reviews,
    load_amazon_reviews,
    quick_inspect
)

config = load_config()
print(f"Config: {config['project']['name']}")

print("\n--- Testing Yelp loader ---")
yelp_df = load_yelp_reviews(
    filepath=config['data']['yelp']['review_file'],
    max_rows=1000
)
quick_inspect(yelp_df)

print("\n--- Testing Amazon loader ---")
amazon_df = load_amazon_reviews(
    filepath="data/raw/All_Beauty.jsonl",
    max_rows=1000
)
quick_inspect(amazon_df)

print("\n--- Testing combined ---")
combined = pd.concat([yelp_df, amazon_df], ignore_index=True)
quick_inspect(combined)

print("Loader test passed.")