import torch
import pandas as pd
import random
from torch.utils.data import Dataset
from src.utils.logger import get_logger

logger = get_logger("pretrain_dataset")


class PretrainDataset(Dataset):

    def __init__(
        self,
        csv_path: str,
        tokenizer,
        max_seq_length: int = 512,
        mask_probability: float = 0.15
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.mask_probability = mask_probability

        self.mask_id = tokenizer.special_tokens["<MASK>"]
        self.pad_id = tokenizer.special_tokens["<PAD>"]
        self.unk_id = tokenizer.special_tokens["<UNK>"]

        logger.info(f"Loading pretrain data from {csv_path}")
        df = pd.read_csv(csv_path)
        self.texts = df["text"].tolist()
        logger.info(f"Loaded {len(self.texts):,} texts for pretraining")

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        text = self.texts[idx]

        token_ids = self.tokenizer.encode(
            text,
            max_length=self.max_seq_length
        )
        token_ids = self.tokenizer.pad_sequence(
            token_ids,
            self.max_seq_length
        )

        input_ids, labels = self._mask_tokens(token_ids)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels":    torch.tensor(labels,    dtype=torch.long)
        }

    def _mask_tokens(self, token_ids: list) -> tuple:
        input_ids = token_ids.copy()
        labels = [-100] * len(token_ids)

        special_ids = set(self.tokenizer.special_tokens.values())

        for i, token_id in enumerate(token_ids):
            if token_id in special_ids:
                continue
            if token_id == self.pad_id:
                continue

            if random.random() < self.mask_probability:
                labels[i] = token_id

                roll = random.random()
                if roll < 0.80:
                    input_ids[i] = self.mask_id
                elif roll < 0.90:
                    input_ids[i] = random.randint(
                        len(special_ids),
                        len(self.tokenizer.token2id) - 1
                    )

        return input_ids, labels