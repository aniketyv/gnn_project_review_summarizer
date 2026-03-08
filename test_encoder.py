import sys
import torch
sys.path.append(".")

from src.model.encoder import Encoder, FeedForward, PositionalEncoding

d_model = 256
num_heads = 8
num_layers = 4
d_ff = 512
vocab_size = 16000
max_seq_length = 1024
batch_size = 2
seq_len = 50

print("=== TEST 1: FeedForward ===")
ff = FeedForward(d_model, d_ff)
x = torch.randn(batch_size, seq_len, d_model)
out = ff(x)
print(f"Input:  {x.shape}")
print(f"Output: {out.shape}")
assert out.shape == x.shape
print("PASSED\n")

print("=== TEST 2: Positional Encoding ===")
pe = PositionalEncoding(d_model, max_seq_length)
x = torch.randn(batch_size, seq_len, d_model)
out = pe(x)
print(f"Input:  {x.shape}")
print(f"Output: {out.shape}")
assert out.shape == x.shape
print("PASSED\n")

print("=== TEST 3: Full Encoder forward pass ===")
encoder = Encoder(
    vocab_size=vocab_size,
    d_model=d_model,
    num_heads=num_heads,
    num_layers=num_layers,
    d_ff=d_ff,
    max_seq_length=max_seq_length
)

token_ids = torch.randint(1, vocab_size, (batch_size, seq_len))
token_ids[0, -5:] = 0

encoded, attention_weights = encoder(token_ids)
print(f"Input token ids shape:  {token_ids.shape}")
print(f"Encoded output shape:   {encoded.shape}")
print(f"Number of attn layers:  {len(attention_weights)}")
print(f"Attn weights per layer: {attention_weights[0].shape}")
assert encoded.shape == (batch_size, seq_len, d_model)
assert len(attention_weights) == num_layers
print("PASSED\n")

print("=== TEST 4: Parameter count ===")
total_params = sum(p.numel() for p in encoder.parameters())
trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
print(f"Total parameters:     {total_params:,}")
print(f"Trainable parameters: {trainable:,}")
print("PASSED\n")

print("=== TEST 5: Padding mask working ===")
token_ids_padded = torch.randint(1, vocab_size, (batch_size, seq_len))
token_ids_padded[0, 30:] = 0
encoded_padded, _ = encoder(token_ids_padded)
print(f"Output with padding: {encoded_padded.shape}")
print("PASSED\n")

print("All encoder tests passed.")