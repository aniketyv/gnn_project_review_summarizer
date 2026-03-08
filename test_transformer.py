import sys
import torch
sys.path.append(".")

from src.model.transformer import Transformer
from src.tokenizer.bpe_tokenizer import BPETokenizer
from src.data.loader import load_config

config = load_config()

print("=== TEST 1: Model initialization ===")
model = Transformer(config)
print("PASSED\n")

print("=== TEST 2: Forward pass ===")
batch_size = 2
src_len = 50
tgt_len = 20
vocab_size = config["tokenizer"]["vocab_size"]

src = torch.randint(1, vocab_size, (batch_size, src_len))
tgt = torch.randint(1, vocab_size, (batch_size, tgt_len))

logits, enc_attn, dec_self_attn, dec_cross_attn = model(src, tgt)

print(f"Source shape:        {src.shape}")
print(f"Target shape:        {tgt.shape}")
print(f"Logits shape:        {logits.shape}")
print(f"Encoder attn layers: {len(enc_attn)}")
print(f"Decoder self layers: {len(dec_self_attn)}")
print(f"Decoder cross layers:{len(dec_cross_attn)}")
assert logits.shape == (batch_size, tgt_len, vocab_size)
print("PASSED\n")

print("=== TEST 3: Loss computation ===")
criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
logits_flat = logits.reshape(-1, vocab_size)
tgt_flat = tgt.reshape(-1)
loss = criterion(logits_flat, tgt_flat)
print(f"Loss value: {loss.item():.4f}")
print(f"Expected ~log({vocab_size}) = {torch.log(torch.tensor(float(vocab_size))):.4f}")
assert loss.item() > 0
print("PASSED\n")

print("=== TEST 4: Generate function ===")
tokenizer = BPETokenizer()
tokenizer.load("data/processed/tokenizer.json")

src_single = torch.randint(1, vocab_size, (1, 50))
summaries = model.generate(
    src_single,
    tokenizer,
    max_length=30,
    temperature=0.7
)
print(f"Generated summary: '{summaries[0]}'")
print(f"Summary length: {len(summaries[0].split())} words")
print("PASSED\n")

print("=== TEST 5: Masks shape ===")
src_mask = model.make_src_padding_mask(src)
tgt_mask = model.make_tgt_mask(tgt)
print(f"Source mask shape: {src_mask.shape}")
print(f"Target mask shape: {tgt_mask.shape}")
print("PASSED\n")

print("=== TEST 6: Device move ===")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)
src = src.to(device)
tgt = tgt.to(device)
logits, _, _, _ = model(src, tgt)
print(f"Model on device: {device}")
print(f"Logits device:   {logits.device}")
print("PASSED\n")

print("All transformer tests passed.")