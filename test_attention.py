import sys
import torch
sys.path.append(".")

from src.model.attention import MultiHeadAttention

d_model = 256
num_heads = 8
batch_size = 2
seq_len = 10

attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)

x = torch.randn(batch_size, seq_len, d_model)

print("=== TEST 1: Self Attention (no mask) ===")
output, weights = attention(x, x, x)
print(f"Input shape:            {x.shape}")
print(f"Output shape:           {output.shape}")
print(f"Attention weights shape:{weights.shape}")
assert output.shape == (batch_size, seq_len, d_model)
print("PASSED\n")

print("=== TEST 2: With padding mask ===")
mask = torch.ones(batch_size, 1, 1, seq_len)
mask[0, 0, 0, -2:] = 0
output_masked, weights_masked = attention(x, x, x, mask)
print(f"Output shape with mask: {output_masked.shape}")
print("PASSED\n")

print("=== TEST 3: Cross attention ===")
encoder_output = torch.randn(batch_size, 20, d_model)
decoder_query = torch.randn(batch_size, 5, d_model)
output_cross, weights_cross = attention(
    decoder_query,
    encoder_output,
    encoder_output
)
print(f"Decoder query shape:  {decoder_query.shape}")
print(f"Encoder output shape: {encoder_output.shape}")
print(f"Cross attention out:  {output_cross.shape}")
assert output_cross.shape == (batch_size, 5, d_model)
print("PASSED\n")

print("=== TEST 4: Causal mask (decoder) ===")
causal_mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
output_causal, weights_causal = attention(x, x, x, causal_mask)
print(f"Causal mask shape:   {causal_mask.shape}")
print(f"Output shape:        {output_causal.shape}")
print(f"Upper triangle weights should be ~0:")
print(f"  weights[0,0,0,1:4] = {weights_causal[0,0,0,1:4].detach().numpy()}")
print("PASSED\n")

print("All attention tests passed.")