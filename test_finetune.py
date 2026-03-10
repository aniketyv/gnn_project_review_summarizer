import sys
import torch
sys.path.append(".")

from src.data.loader import load_config
from src.tokenizer.bpe_tokenizer import BPETokenizer
from src.data.finetune_dataset import FinetuneDataset
from torch.utils.data import DataLoader

config = load_config()

tokenizer = BPETokenizer()
tokenizer.load("data/processed/tokenizer.json")

print("=== TEST 1: FinetuneDataset ===")
dataset = FinetuneDataset(
    csv_path="data/processed/train.csv",
    tokenizer=tokenizer,
    src_max_length=256,
    tgt_max_length=64,
    min_reviews_per_business=3
)
print(f"Number of pairs: {len(dataset):,}")

sample = dataset[0]
print(f"src_ids shape:   {sample['src_ids'].shape}")
print(f"tgt_ids shape:   {sample['tgt_ids'].shape}")
print(f"label_ids shape: {sample['label_ids'].shape}")
print("PASSED\n")

print("=== TEST 2: Decode a sample ===")
src_text = tokenizer.decode(
    sample["src_ids"].tolist()
)
tgt_text = tokenizer.decode(
    [t for t in sample["tgt_ids"].tolist() if t not in
     tokenizer.special_tokens.values()]
)
print(f"Input (first 100 chars):   {src_text[:100]}")
print(f"Summary (first 100 chars): {tgt_text[:100]}")
print("PASSED\n")

print("=== TEST 3: DataLoader ===")
loader = DataLoader(dataset, batch_size=4, shuffle=True)
batch = next(iter(loader))
print(f"Batch src shape:   {batch['src_ids'].shape}")
print(f"Batch tgt shape:   {batch['tgt_ids'].shape}")
print(f"Batch label shape: {batch['label_ids'].shape}")
print("PASSED\n")

print("=== TEST 4: Forward pass with pretrained model ===")
from src.training.finetune import load_pretrained_model
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = load_pretrained_model(
    config,
    device,
    "checkpoints/pretrain/checkpoint_epoch_10.pt"
)

src = batch["src_ids"].to(device)
tgt = batch["tgt_ids"].to(device)
labels = batch["label_ids"].to(device)

import torch.nn as nn
criterion = nn.CrossEntropyLoss(ignore_index=-100)

logits, _, _, _ = model(src, tgt)
loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
print(f"Finetune forward loss: {loss.item():.4f}")
print("PASSED\n")

print("All finetune tests passed. Ready to run finetuning.")
print("\nTo start finetuning run:")
print("python main.py --mode finetune")