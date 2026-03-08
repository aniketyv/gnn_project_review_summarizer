import torch
import torch.nn as nn
import math
from src.model.attention import MultiHeadAttention
from src.model.encoder import FeedForward, PositionalEncoding


class DecoderLayer(nn.Module):

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        self.masked_self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor = None, # type: ignore
        tgt_mask: torch.Tensor = None # type: ignore
    ) -> tuple:

        normed = self.norm1(x)
        attended, self_attn_weights = self.masked_self_attention(
            normed, normed, normed, tgt_mask
        )
        x = x + self.dropout(attended)

        normed = self.norm2(x)
        crossed, cross_attn_weights = self.cross_attention(
            normed, encoder_output, encoder_output, src_mask
        )
        x = x + self.dropout(crossed)

        normed = self.norm3(x)
        forwarded = self.feed_forward(normed)
        x = x + self.dropout(forwarded)

        return x, self_attn_weights, cross_attn_weights


class Decoder(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int,
        max_seq_length: int,
        dropout: float = 0.1
    ):
        super().__init__()

        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.positional_encoding = PositionalEncoding(
            d_model, max_seq_length, dropout
        )
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)

    def make_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor = None, # type: ignore
        tgt_mask: torch.Tensor = None # type: ignore
    ) -> tuple:

        if tgt_mask is None:
            tgt_mask = self.make_causal_mask(x.size(1), x.device)

        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)

        all_self_attn_weights = []
        all_cross_attn_weights = []

        for layer in self.layers:
            x, self_attn_w, cross_attn_w = layer(
                x, encoder_output, src_mask, tgt_mask
            )
            all_self_attn_weights.append(self_attn_w)
            all_cross_attn_weights.append(cross_attn_w)

        x = self.norm(x)
        logits = self.output_projection(x)

        return logits, all_self_attn_weights, all_cross_attn_weights