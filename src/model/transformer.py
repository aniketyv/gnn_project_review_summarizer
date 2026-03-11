import torch
import torch.nn as nn
from src.model.encoder import Encoder
from src.model.decoder import Decoder
from src.utils.logger import get_logger

logger = get_logger("transformer")


class Transformer(nn.Module):

    def __init__(self, config: dict):
        super().__init__()

        vocab_size = config["tokenizer"]["vocab_size"]
        d_model = config["model"]["d_model"]
        num_heads = config["model"]["num_heads"]
        num_encoder_layers = config["model"]["num_encoder_layers"]
        num_decoder_layers = config["model"]["num_decoder_layers"]
        d_ff = config["model"]["d_ff"]
        max_seq_length = config["model"]["max_seq_length"]
        dropout = config["model"]["dropout"]

        self.encoder = Encoder(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_encoder_layers,
            d_ff=d_ff,
            max_seq_length=max_seq_length,
            dropout=dropout
        )

        self.decoder = Decoder(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_decoder_layers,
            d_ff=d_ff,
            max_seq_length=max_seq_length,
            dropout=dropout
        )

        self._init_weights()

        total_params = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Transformer initialized")
        logger.info(f"Total parameters:     {total_params:,}")
        logger.info(f"Trainable parameters: {trainable:,}")

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_src_padding_mask(self, src: torch.Tensor) -> torch.Tensor:
        return (src != 0).unsqueeze(1).unsqueeze(2)

    def make_tgt_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        tgt_len = tgt.size(1)
        tgt_padding_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)
        causal_mask = torch.tril(
            torch.ones(tgt_len, tgt_len, device=tgt.device)
        ).unsqueeze(0).unsqueeze(0)
        return tgt_padding_mask & causal_mask.bool()

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor = None, # type: ignore
        tgt_mask: torch.Tensor = None  # type: ignore
    ) -> tuple:

        if src_mask is None:
            src_mask = self.make_src_padding_mask(src)
        if tgt_mask is None:
            tgt_mask = self.make_tgt_mask(tgt)

        encoder_output, encoder_attn = self.encoder(src, src_mask)

        logits, decoder_self_attn, decoder_cross_attn = self.decoder(
            tgt, encoder_output, src_mask, tgt_mask
        )

        return logits, encoder_attn, decoder_self_attn, decoder_cross_attn

    def generate(
        self,
        src: torch.Tensor,
        tokenizer,
        max_length: int = 80,
        temperature: float = 0.7,
        top_p: float = 0.92
    ) -> list:
        self.eval()
        device = src.device
        batch_size = src.size(0)

        src_mask = self.make_src_padding_mask(src)
        encoder_output, _ = self.encoder(src, src_mask)

        bos_id = tokenizer.special_tokens["<BOS>"]
        eos_id = tokenizer.special_tokens["<EOS>"]
        pad_id = tokenizer.special_tokens["<PAD>"]

        generated = torch.full(
            (batch_size, 1),
            bos_id,
            dtype=torch.long,
            device=device
        )

        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        with torch.no_grad():
            for step in range(max_length):
                tgt_mask = self.make_tgt_mask(generated)

                logits, _, _ = self.decoder(
                    generated, encoder_output, src_mask, tgt_mask
                )

                next_token_logits = logits[:, -1, :]
                next_token_logits = next_token_logits / temperature

                # Our model was generating reviews based on positive reviews 
                
                # Repetition penalty — divide logits of already generated tokens
                # Makes model less likely to repeat words it already used
                for b in range(batch_size):
                    for token_id in set(generated[b].tolist()):
                        next_token_logits[b, token_id] /= 1.3

                next_token_logits[:, pad_id] = -1e9

                probs = torch.softmax(next_token_logits, dim=-1)

                sorted_probs, sorted_indices = torch.sort(
                    probs, dim=-1, descending=True
                )
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs - sorted_probs > top_p
                sorted_probs[sorted_indices_to_remove] = 0
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

                next_tokens = torch.multinomial(sorted_probs, num_samples=1)
                next_tokens = sorted_indices.gather(-1, next_tokens)

                next_tokens[finished] = pad_id
                generated = torch.cat([generated, next_tokens], dim=1)

                finished = finished | (next_tokens.squeeze(-1) == eos_id)
                if finished.all():
                    break

        summaries = []
        for i in range(batch_size):
            tokens = generated[i].tolist()
            if eos_id in tokens:
                tokens = tokens[:tokens.index(eos_id)]
            tokens = tokens[1:]
            summaries.append(tokenizer.decode(tokens))

        return summaries