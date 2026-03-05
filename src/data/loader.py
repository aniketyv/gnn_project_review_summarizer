# load_config – Reads a settings file (in YAML format) and turns it into a dictionary.
# load_yelp_reviews – Loads Yelp and Amazon review data from a file where each line is a separate review (JSON format).

import json
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import yaml
from src.utils.logger import get_logger

logger = get_logger("loader")


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_yelp_reviews(filepath: str, max_rows: int = 100000) -> pd.DataFrame:
    logger.info(f"Loading Yelp reviews from: {filepath}")
    records = []
    skipped = 0

    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(tqdm(f, desc="Loading Yelp")):
            if i >= max_rows:
                break
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                records.append({
                    "review_id":   r.get("review_id", ""),
                    "user_id":     r.get("user_id", ""),
                    "business_id": r.get("business_id", ""),
                    "stars":       float(r.get("stars", 0)),
                    "useful":      int(r.get("useful", 0)),
                    "text":        r.get("text", ""),
                    "date":        r.get("date", ""),
                    "source":      "yelp"
                })
            except (json.JSONDecodeError, ValueError):
                skipped += 1
                continue

    df = pd.DataFrame(records)
    logger.info(f"Yelp loaded: {len(df):,} reviews | skipped: {skipped:,}")
    return df


def load_amazon_reviews(filepath: str, max_rows: int = 50000) -> pd.DataFrame:
    logger.info(f"Loading Amazon reviews from: {filepath}")
    records = []
    skipped = 0

    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(tqdm(f, desc="Loading Amazon")):
            if i >= max_rows:
                break
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)

                # Convert timestamp (milliseconds) to date string
                # Amazon uses Unix timestamp in milliseconds
                # Yelp uses "YYYY-MM-DD HH:MM:SS" string
                # We normalize to same format
                ts = r.get("timestamp", 0)
                if ts:
                    date_str = datetime.fromtimestamp(
                        ts / 1000
                    ).strftime("%Y-%m-%d %H:%M:%S")
                else:
                    date_str = ""

                # Combine title + text for richer review content
                # Amazon has separate title field, Yelp does not
                # Merging gives model more content to learn from
                title = r.get("title", "").strip()
                text = r.get("text", "").strip()
                full_text = f"{title}. {text}" if title else text

                records.append({
                    "review_id":   r.get("asin", "") + "_" + r.get("user_id", ""),
                    "user_id":     r.get("user_id", ""),
                    "business_id": r.get("parent_asin", ""),
                    "stars":       float(r.get("rating", 0)),
                    "useful":      int(r.get("helpful_vote", 0)),
                    "text":        full_text,
                    "date":        date_str,
                    "source":      "amazon"
                })
            except (json.JSONDecodeError, ValueError):
                skipped += 1
                continue

    df = pd.DataFrame(records)
    logger.info(f"Amazon loaded: {len(df):,} reviews | skipped: {skipped:,}")
    return df


def load_all_reviews(config: dict) -> pd.DataFrame:
    """
    Loads both Yelp and Amazon datasets and combines
    them into one unified DataFrame.
    This is the main function the rest of the project calls.
    """
    yelp_config = config["data"]["yelp"]
    amazon_config = config["data"]["amazon"]

    # Load Yelp
    yelp_df = load_yelp_reviews(
        filepath=yelp_config["review_file"],
        max_rows=yelp_config["max_rows"]
    )

    # Load each Amazon category
    amazon_dfs = []
    for category_file in amazon_config["review_files"]:
        amazon_df = load_amazon_reviews(
            filepath=category_file,
            max_rows=amazon_config["max_rows_per_category"]
        )
        amazon_dfs.append(amazon_df)

    # Combine everything
    all_dfs = [yelp_df] + amazon_dfs
    combined = pd.concat(all_dfs, ignore_index=True)

    logger.info(f"Total combined reviews: {len(combined):,}")
    logger.info(f"Yelp reviews: {(combined['source']=='yelp').sum():,}")
    logger.info(f"Amazon reviews: {(combined['source']=='amazon').sum():,}")

    return combined


def quick_inspect(df: pd.DataFrame) -> None:
    print("\n" + "="*50)
    print("DATASET INSPECTION")
    print("="*50)
    print(f"Total reviews:       {len(df):,}")
    print(f"Unique businesses:   {df['business_id'].nunique():,}")
    print(f"Unique users:        {df['user_id'].nunique():,}")
    print(f"Avg review length:   {df['text'].str.len().mean():.0f} chars")
    print(f"Min review length:   {df['text'].str.len().min()} chars")
    print(f"Max review length:   {df['text'].str.len().max()} chars")
    print(f"\nStars distribution:")
    stars = df["stars"].value_counts().sort_index()
    for star, count in stars.items():
        bar = "█" * int(count / stars.max() * 20)
        print(f"  {star}★  {bar} {count:,}")
    print(f"\nUseful votes > 0:    {(df['useful'] > 0).sum():,}")
    if "source" in df.columns:
        print(f"\nSource breakdown:")
        for source, count in df["source"].value_counts().items():
            print(f"  {source}: {count:,}")
    print(f"\nSample review:")
    print(f"  {df['text'].iloc[0][:200]}...")
    print("="*50 + "\n")