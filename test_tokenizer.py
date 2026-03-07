import sys
import pandas as pd
sys.path.append(".")

from src.tokenizer.bpe_tokenizer import BPETokenizer
from src.data.loader import load_config

config = load_config()

# Load processed training data
train_df = pd.read_csv("data/processed/train.csv")
texts = train_df["text"].tolist()
print(f"Training tokenizer on {len(texts):,} reviews")

# Train tokenizer
tokenizer = BPETokenizer(vocab_size=config["tokenizer"]["vocab_size"])
tokenizer.train(texts)

# Save it
tokenizer.save("data/processed/tokenizer.json")

# Test encoding and decoding
test_sentences = [
    "the food was absolutely amazing and the service was great",
    "terrible experience the waiter was rude and food was cold",
    "best restaurant i have ever visited in my entire life"
]

print("\n=== ENCODING TEST ===")
for sentence in test_sentences:
    ids = tokenizer.encode(sentence)
    decoded = tokenizer.decode(ids)
    print(f"Original:  {sentence}")
    print(f"Token ids: {ids[:10]}... ({len(ids)} tokens)")
    print(f"Decoded:   {decoded}")
    print()

# Test special tokens
print("=== SPECIAL TOKENS TEST ===")
ids = tokenizer.encode(
    "great food and service",
    add_special_tokens=True
)
print(f"With special tokens: {ids}")
print(f"BOS id: {tokenizer.special_tokens['<BOS>']}")
print(f"EOS id: {tokenizer.special_tokens['<EOS>']}")

# Test padding
print("\n=== PADDING TEST ===")
ids = tokenizer.encode("good food")
padded = tokenizer.pad_sequence(ids, max_length=20)
print(f"Original length: {len(ids)}")
print(f"Padded length:   {len(padded)}")
print(f"Padded sequence: {padded}")

print("\nTokenizer test passed.")