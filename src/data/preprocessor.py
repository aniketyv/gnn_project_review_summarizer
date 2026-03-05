# According to us - a good review is the one which is :-

# 1) not short
# 2) doesnt contains any Phonenumber / emaill/ and url. ( review should be descriptive not one which is being used for advertisement)
# 3) For the moment it should be in meaningfull english.

import re
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger("preprocessor")


def remove_nulls(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.dropna(subset=["text"])
    df = df[df["text"].str.strip() != ""]
    after = len(df)
    logger.info(f"Remove nulls: {before:,} → {after:,} (removed {before-after:,})")
    return df


def remove_short_reviews(df: pd.DataFrame, min_length: int = 50) -> pd.DataFrame:
    before = len(df)
    df = df[df["text"].str.len() >= min_length]
    after = len(df)
    logger.info(f"Remove short reviews: {before:,} → {after:,} (removed {before-after:,})")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset=["text"])
    after = len(df)
    logger.info(f"Remove duplicates: {before:,} → {after:,} (removed {before-after:,})")
    return df


def clean_text(text: str) -> str:
    # Remove HTML entities
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"<[^>]+>", "", text)

    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", "", text)

    # Remove phone numbers
    text = re.sub(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "", text)

    # Remove non-ASCII characters
    text = text.encode("ascii", "ignore").decode("ascii")

    # Normalize whitespace
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    # Lowercase
    text = text.lower()

    return text


def apply_text_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Cleaning text...")
    df = df.copy()
    df["text"] = df["text"].apply(clean_text)
    logger.info("Text cleaning done")
    return df


def filter_english(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)

    def is_english(text: str) -> bool:
        try:
            # Simple heuristic: if more than 80% of characters
            # are ASCII letters the review is likely English
            ascii_chars = sum(1 for c in text if c.isascii() and c.isalpha())
            total_chars = sum(1 for c in text if c.isalpha())
            if total_chars == 0:
                return False
            return (ascii_chars / total_chars) >= 0.8
        except Exception:
            return False

    df = df[df["text"].apply(is_english)]
    after = len(df)
    logger.info(f"Filter English: {before:,} → {after:,} (removed {before-after:,})")
    return df


def preprocess(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Master function that runs all cleaning steps in order.
    This is the only function the rest of the project calls.
    """
    logger.info(f"Starting preprocessing on {len(df):,} reviews")
    logger.info(f"Source breakdown before: {df['source'].value_counts().to_dict()}")

    min_length = config["data"]["yelp"]["min_review_length"]

    df = remove_nulls(df)
    df = remove_short_reviews(df, min_length=min_length)
    df = remove_duplicates(df)
    df = apply_text_cleaning(df)
    df = filter_english(df)

    # Reset index after all filtering
    df = df.reset_index(drop=True)

    logger.info(f"Preprocessing complete: {len(df):,} reviews remaining")
    logger.info(f"Source breakdown after: {df['source'].value_counts().to_dict()}")

    return df

# Important: we split by business_id not by individual review.
#     Why? If reviews from same business appear in both train and test,
#     the model might memorize that business instead of learning
#     to generalize. Splitting by business prevents this.

def split_data(df: pd.DataFrame, config: dict) -> tuple:

    train_ratio = config["data"]["yelp"]["train_split"]
    val_ratio = config["data"]["yelp"]["val_split"]

    # Get unique businesses
    businesses = df["business_id"].unique().tolist()
    n = len(businesses)

    # Shuffle businesses
    import numpy as np
    np.random.seed(42)
    np.random.shuffle(businesses)

    # Split business ids
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_businesses = businesses[:train_end]
    val_businesses = businesses[train_end:val_end]
    test_businesses = businesses[val_end:]

    # Split dataframe based on business ids
    train_df = df[df["business_id"].isin(train_businesses)].reset_index(drop=True)
    val_df = df[df["business_id"].isin(val_businesses)].reset_index(drop=True)
    test_df = df[df["business_id"].isin(test_businesses)].reset_index(drop=True)

    logger.info(f"Train: {len(train_df):,} reviews ({len(train_businesses):,} businesses)")
    logger.info(f"Val:   {len(val_df):,} reviews ({len(val_businesses):,} businesses)")
    logger.info(f"Test:  {len(test_df):,} reviews ({len(test_businesses):,} businesses)")

    return train_df, val_df, test_df


def save_splits(train_df, val_df, test_df, config: dict) -> None:
    """
    Saves the three splits to disk as CSV files.
    We save once here and never reprocess again.
    """
    import os
    processed_dir = config["data"]["processed_dir"]
    os.makedirs(processed_dir, exist_ok=True)

    train_path = f"{processed_dir}/train.csv"
    val_path = f"{processed_dir}/val.csv"
    test_path = f"{processed_dir}/test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    logger.info(f"Saved train → {train_path}")
    logger.info(f"Saved val   → {val_path}")
    logger.info(f"Saved test  → {test_path}")