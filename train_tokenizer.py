from src.tokenizer.bpe_tokenizer import BPETokenizer
from src.data.loader import load_config
import pandas as pd

config = load_config()

print("Loading training data...")
df = pd.read_csv("data/processed/train.csv")
texts = df["text"].dropna().tolist()
print(f"Training tokenizer on {len(texts):,} texts...")

tokenizer = BPETokenizer()
tokenizer.train(texts)
tokenizer.save("data/processed/tokenizer.json")

print(f"Done. Vocabulary size: {tokenizer.vocab_size:,}")
print("Saved to data/processed/tokenizer.json")