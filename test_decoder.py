import sys
import torch
sys.path.append(".")

from src.model.encoder import Encoder
from src.model.decoder import Decoder

d_model = 256
num_heads = 8
num_layers = 4
d_ff = 512
vocab_size = 16000
max_seq_length = 1024
batch_size = 2
src_seq_len = 50
tgt_seq_len = 20

encoder = Encoder(
    vocab_size=vocab_size,
    d_model=d_model,
    num_heads=num_heads,
    num_layers=num_layers,
    d_ff=d_ff,
    max_seq_length=max_seq_length
)

decoder = Decoder(
    vocab_size=vocab_size,
    d_model=d_model,
    num_heads=num_heads,
    num_layers=num_layers,
    d_ff=d_ff,
    max_seq_length=max_seq_length
)

print("=== TEST 1: Encoder forward pass ===")
src_tokens = torch.randint(1, vocab_size, (batch_size, src_seq_len))
encoder_output, _ = encoder(src_tokens)
print(f"Encoder output: {encoder_output.shape}")
assert encoder_output.shape == (batch_size, src_seq_len, d_model)
print("PASSED\n")

print("=== TEST 2: Decoder forward pass ===")
tgt_tokens = torch.randint(1, vocab_size, (batch_size, tgt_seq_len))
logits, self_attn_w, cross_attn_w = decoder(tgt_tokens, encoder_output)
print(f"Target tokens:        {tgt_tokens.shape}")
print(f"Logits shape:         {logits.shape}")
print(f"Self attn layers:     {len(self_attn_w)}")
print(f"Cross attn layers:    {len(cross_attn_w)}")
assert logits.shape == (batch_size, tgt_seq_len, vocab_size)
print("PASSED\n")

print("=== TEST 3: Logits to probabilities ===")
probs = torch.softmax(logits[0, 0, :], dim=-1)
print(f"Probabilities sum: {probs.sum().item():.6f}")
print(f"Max probability:   {probs.max().item():.6f}")
print(f"Predicted token:   {probs.argmax().item()}")
assert abs(probs.sum().item() - 1.0) < 1e-5
print("PASSED\n")

print("=== TEST 4: Causal mask shape ===")
causal_mask = decoder.make_causal_mask(tgt_seq_len, torch.device("cpu"))
print(f"Causal mask shape: {causal_mask.shape}")
print(f"Mask preview (5x5):\n{causal_mask[0,0,:5,:5]}")
assert causal_mask.shape == (1, 1, tgt_seq_len, tgt_seq_len)
print("PASSED\n")

print("=== TEST 5: Parameter count ===")
enc_params = sum(p.numel() for p in encoder.parameters())
dec_params = sum(p.numel() for p in decoder.parameters())
total = enc_params + dec_params
print(f"Encoder parameters: {enc_params:,}")
print(f"Decoder parameters: {dec_params:,}")
print(f"Total parameters:   {total:,}")
print("PASSED\n")

print("=== TEST 6: Cross attention shape ===")
print(f"Cross attn weights shape: {cross_attn_w[0].shape}")
print(f"  batch={cross_attn_w[0].shape[0]}")
print(f"  heads={cross_attn_w[0].shape[1]}")
print(f"  tgt_len={cross_attn_w[0].shape[2]}")
print(f"  src_len={cross_attn_w[0].shape[3]}")
assert cross_attn_w[0].shape == (batch_size, num_heads, tgt_seq_len, src_seq_len)
print("PASSED\n")

print("All decoder tests passed.")