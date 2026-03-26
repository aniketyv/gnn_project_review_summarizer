from src.data.loader import load_config, load_all_reviews
from src.data.preprocessor import preprocess, split_data, save_splits

config = load_config()

print("Loading all reviews...")
df = load_all_reviews(config)

print("Preprocessing...")
df = preprocess(df, config)

print("Splitting and saving...")
train_df, val_df, test_df = split_data(df, config)
save_splits(train_df, val_df, test_df, config)

print(f"Done. Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")