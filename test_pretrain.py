import sys
import torch
sys.path.append(".")

from src.data.loader import load_config
from src.tokenizer.bpe_tokenizer import BPETokenizer
from src.data.pretrain_dataset import PretrainDataset
from torch.utils.data import DataLoader

config = load_config()

tokenizer = BPETokenizer()
tokenizer.load("data/processed/tokenizer.json")

print("=== TEST 1: PretrainDataset ===")
dataset = PretrainDataset(
    csv_path="data/processed/train.csv",
    tokenizer=tokenizer,
    max_seq_length=128,
    mask_probability=0.15
)
print(f"Dataset size: {len(dataset):,}")

sample = dataset[0]
print(f"Input ids shape: {sample['input_ids'].shape}")
print(f"Labels shape:    {sample['labels'].shape}")

masked_count = (sample['labels'] != -100).sum().item()
total_count = (sample['input_ids'] != 0).sum().item()
print(f"Masked tokens:   {masked_count}/{total_count} ({masked_count/total_count*100:.1f}%)")
print("PASSED\n")

print("=== TEST 2: DataLoader ===")
loader = DataLoader(dataset, batch_size=4, shuffle=True)
batch = next(iter(loader))
print(f"Batch input shape: {batch['input_ids'].shape}")
print(f"Batch labels shape:{batch['labels'].shape}")
print("PASSED\n")

print("=== TEST 3: Quick forward pass ===")
from src.model.transformer import Transformer
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = Transformer(config).to(device)

input_ids = batch["input_ids"].to(device)
labels = batch["labels"].to(device)

import torch.nn as nn
criterion = nn.CrossEntropyLoss(ignore_index=-100)

logits, _, _, _ = model(input_ids, input_ids)
loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
print(f"Forward pass loss: {loss.item():.4f}")
print("PASSED\n")

print("All pretrain tests passed. Ready to run pretraining.")
print("\nTo start pretraining run:")
print("python main.py --mode pretrain")