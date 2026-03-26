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
        src_mask: torch.Tensor = None,  # type: ignore
        tgt_mask: torch.Tensor = None   # type: ignore
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
        max_length:         int   = 80,
        temperature:        float = 0.7,
        top_p:              float = 0.92,
        beam_size:          int   = 4,
        repetition_penalty: float = 1.3
    ) -> list:
        # Fix - along with repetation penalty have added beamsearch 
        self.eval()
        device = src.device
        batch_size = src.size(0)

        bos_id = tokenizer.special_tokens["<BOS>"]
        eos_id = tokenizer.special_tokens["<EOS>"]
        pad_id = tokenizer.special_tokens["<PAD>"]

        # Encode source once — reuse for all beams
        src_mask = self.make_src_padding_mask(src)
        encoder_output, _ = self.encoder(src, src_mask)

        results = []

        with torch.no_grad():
            for b in range(batch_size):

                # Single item encoder output
                enc_out  = encoder_output[b].unsqueeze(0)   # (1, src_len, d)
                src_msk  = src_mask[b].unsqueeze(0)         # (1, 1, 1, src_len)

                # Each beam is (cumulative_log_prob, token_ids_list)
                beams     = [(0.0, [bos_id])]
                completed = []

                for _ in range(max_length):
                    new_beams = []

                    for score, tokens in beams:

                        # If beam already ended just carry it forward
                        if tokens[-1] == eos_id:
                            completed.append((score, tokens))
                            continue

                        tgt = torch.tensor(
                            [tokens], dtype=torch.long
                        ).to(device)

                        tgt_mask = self.make_tgt_mask(tgt)

                        logits, _, _ = self.decoder(
                            tgt, enc_out, src_msk, tgt_mask
                        )

                        # Take logits for last position only
                        next_logits = logits[0, -1, :]  # (vocab_size,)

                        # Repetition penalty
                        # Divides logit of any token already in sequence
                        # Positive logits shrink, negative logits grow more negative
                        for token_id in set(tokens):
                            if next_logits[token_id] > 0:
                                next_logits[token_id] /= repetition_penalty
                            else:
                                next_logits[token_id] *= repetition_penalty

                        # Block PAD token
                        next_logits[pad_id] = -1e9

                        # Temperature scaling
                        next_logits = next_logits / temperature

                        # Top-p nucleus filtering
                        sorted_logits, sorted_idx = torch.sort(
                            next_logits, descending=True
                        )
                        cumulative_probs = torch.cumsum(
                            torch.softmax(sorted_logits, dim=-1), dim=-1
                        )
                        # Remove tokens beyond top_p threshold
                        sorted_logits[cumulative_probs > top_p] = float("-inf")
                        next_logits[sorted_idx] = sorted_logits

                        # Convert to log probs for stable scoring
                        log_probs = torch.log_softmax(next_logits, dim=-1)

                        # Expand top beam_size candidates
                        top_log_probs, top_tokens = torch.topk(
                            log_probs, beam_size
                        )

                        for log_prob, token in zip(
                            top_log_probs.tolist(),
                            top_tokens.tolist()
                        ):
                            new_score = score + log_prob
                            new_beams.append(
                                (new_score, tokens + [token])
                            )

                    if not new_beams:
                        break

                    # Keep only top beam_size beams by score
                    beams = sorted(
                        new_beams, key=lambda x: x[0], reverse=True
                    )[:beam_size]

                    # If all beams ended stop early
                    if all(t[-1] == eos_id for _, t in beams):
                        completed.extend(beams)
                        break

                # Collect all finished and unfinished beams
                all_candidates = completed + beams
                if not all_candidates:
                    results.append("")
                    continue

                # Pick highest scoring sequence
                best_score, best_tokens = max(
                    all_candidates, key=lambda x: x[0]
                )

                # Strip BOS and EOS
                output_ids = best_tokens[1:]  # remove BOS
                if eos_id in output_ids:
                    output_ids = output_ids[:output_ids.index(eos_id)]

                results.append(tokenizer.decode(output_ids))

        return results