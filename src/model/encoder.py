import torch
import torch.nn as nn
import math
from src.model.attention import MultiHeadAttention


class FeedForward(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(
            self.dropout(
                self.activation(self.linear1(x))
            )
        )


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_seq_length: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)] # type: ignore
        return self.dropout(x)


class EncoderLayer(nn.Module):

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> tuple: # type: ignore
        normed = self.norm1(x)
        attended, weights = self.attention(normed, normed, normed, mask)
        x = x + self.dropout(attended)

        normed = self.norm2(x)
        forwarded = self.feed_forward(normed)
        x = x + self.dropout(forwarded)

        return x, weights


class Encoder(nn.Module):

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
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def make_padding_mask(self, x: torch.Tensor) -> torch.Tensor:
        return (x != 0).unsqueeze(1).unsqueeze(2)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> tuple: # type: ignore
        if mask is None:
            mask = self.make_padding_mask(x)

        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)

        all_attention_weights = []
        for layer in self.layers:
            x, weights = layer(x, mask)
            all_attention_weights.append(weights)

        x = self.norm(x)

        return x, all_attention_weights