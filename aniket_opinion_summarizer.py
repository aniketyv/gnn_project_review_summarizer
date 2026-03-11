import sys
import torch
import pandas as pd
sys.path.append(".")

from src.model.transformer import Transformer
from src.tokenizer.bpe_tokenizer import BPETokenizer
from src.data.loader import load_config

def load_model(config, checkpoint_path, device):
    model = Transformer(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    print(f"Model loaded from {checkpoint_path}")
    print(f"Checkpoint val loss: {checkpoint.get('val_loss', 'unknown'):.4f}")
    return model

def get_business_reviews(df, business_id, max_reviews=10):
    reviews = df[df["business_id"] == business_id].copy()
    reviews = reviews.sort_values("useful", ascending=False)
    return reviews.head(max_reviews)

def summarize_business(model, tokenizer, reviews_df, config, device):
    texts = reviews_df["text"].tolist()
    combined = " [SEP] ".join(texts[:8])

    src_ids = tokenizer.encode(
        combined,
        max_length=config["model"]["max_seq_length"]
    )
    src_ids = tokenizer.pad_sequence(
        src_ids,
        config["model"]["max_seq_length"]
    )

    src_tensor = torch.tensor([src_ids], dtype=torch.long).to(device)

    with torch.no_grad():
        summaries = model.generate(
            src_tensor,
            tokenizer,
            max_length=80,
            temperature=0.7,
            top_p=0.92
        )

    return summaries[0]

def run_demo():
    config = load_config()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}\n")

    tokenizer = BPETokenizer()
    tokenizer.load("data/processed/tokenizer.json")

    model = load_model(
        config,
        "checkpoints/finetune/checkpoint_epoch_30.pt",
        device
    )

    print("\nLoading Yelp reviews...")
    df = pd.read_csv("data/processed/test.csv")
    yelp_df = df[df["source"] == "yelp"].copy()

    # Find businesses with enough reviews
    business_counts = yelp_df["business_id"].value_counts()
    good_businesses = business_counts[business_counts >= 5].index.tolist()
    print(f"Businesses with 5+ reviews: {len(good_businesses)}")

    # Run demo on first 5 businesses
    print("\n" + "="*60)
    print("OPINION SUMMARIZER DEMO")
    print("="*60)

    for i, business_id in enumerate(good_businesses[:10]):
        reviews = get_business_reviews(yelp_df, business_id)

        print(f"\nBusiness {i+1}: {business_id}")
        print(f"Number of reviews: {len(reviews)}")
        print(f"Average stars: {reviews['stars'].mean():.1f}")
        print(f"\nSample reviews:")
        for j, row in reviews.head(3).iterrows():
            print(f"  [{row['stars']}★] {row['text'][:100]}...")

        print(f"\nGenerated Summary:")
        summary = summarize_business(model, tokenizer, reviews, config, device)
        print(f"  → {summary}")
        print("-"*60)

if __name__ == "__main__":
    run_demo()