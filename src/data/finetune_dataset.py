import torch
import pandas as pd
import re
from torch.utils.data import Dataset
from src.utils.logger import get_logger

logger = get_logger("finetune_dataset")


class FinetuneDataset(Dataset):

    def __init__(
        self,
        csv_path: str,
        tokenizer,
        src_max_length: int = 256,
        tgt_max_length: int = 64,
        min_reviews_per_business: int = 3
    ):
        self.tokenizer = tokenizer
        self.src_max_length = src_max_length
        self.tgt_max_length = tgt_max_length

        self.pad_id = tokenizer.special_tokens["<PAD>"]
        self.bos_id = tokenizer.special_tokens["<BOS>"]
        self.eos_id = tokenizer.special_tokens["<EOS>"]
        self.sep_id = tokenizer.special_tokens["<SEP>"]

        logger.info(f"Loading finetune data from {csv_path}")
        df = pd.read_csv(csv_path)

        # Only use Yelp data for finetuning
        # Amazon has no business_id grouping
        yelp_df = df[df["source"] == "yelp"].copy()
        logger.info(f"Yelp reviews available: {len(yelp_df):,}")

        self.pairs = self._build_review_summary_pairs(
            yelp_df,
            min_reviews_per_business
        )
        logger.info(
            f"Built {len(self.pairs):,} review-summary pairs "
            f"from {yelp_df['business_id'].nunique():,} businesses"
        )

    def _extract_summary(self, text: str) -> str:
        
        # Extracts first 1-2 sentences from a review
        # to use as a pseudo-summary.
        # Short enough to be a summary.
        # Long enough to have real content.
        
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

        if not sentences:
            return text[:100]

        # Take first 2 sentences max, but cap at 80 characters
        summary = sentences[0]
        if len(sentences) > 1 and len(summary) < 60:
            summary = summary + " " + sentences[1]

        return summary[:200]

    def _build_review_summary_pairs(
        self,
        df: pd.DataFrame,
        min_reviews: int
        ) -> list:
        
    #  FIX(model started overfitting)
    # Groups reviews by business.
    # Each review takes a turn being the summary target.
    # Remaining reviews become the input.
    # This gives us many more training pairs than
        pairs = []

        grouped = df.groupby("business_id")

        for business_id, group in grouped:
            if len(group) < min_reviews:
                continue

            group = group.sort_values("useful", ascending=False)
            reviews = group["text"].tolist()

            for i in range(len(reviews)):
                summary = self._extract_summary(reviews[i])

                others = [reviews[j] for j in range(len(reviews)) if j != i]
                combined_input = " [SEP] ".join(others[:5])

                if len(summary.split()) < 5:
                    continue
                if len(combined_input.split()) < 20:
                    continue

                pairs.append({
                    "input":       combined_input,
                    "summary":     summary,
                    "business_id": business_id
                })

        return pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        pair = self.pairs[idx]

        src_ids = self.tokenizer.encode(
            pair["input"],
            max_length=self.src_max_length
        )
        src_ids = self.tokenizer.pad_sequence(
            src_ids, self.src_max_length
        )

        # Decoder input: BOS + summary tokens
        tgt_ids = self.tokenizer.encode(
            pair["summary"],
            max_length=self.tgt_max_length - 1,
            add_special_tokens=True
        )
        tgt_ids = self.tokenizer.pad_sequence(
            tgt_ids, self.tgt_max_length
        )

        # Labels: summary tokens + EOS, shifted by 1
        # Model predicts next token at each position
        # -100 at PAD positions so loss ignores them
        label_ids = self.tokenizer.encode(
            pair["summary"],
            max_length=self.tgt_max_length - 1
        )
        label_ids = label_ids + [self.eos_id]
        label_ids = label_ids + [-100] * (
            self.tgt_max_length - len(label_ids)
        )
        label_ids = label_ids[:self.tgt_max_length]

        return {
            "src_ids":  torch.tensor(src_ids,   dtype=torch.long),
            "tgt_ids":  torch.tensor(tgt_ids,   dtype=torch.long),
            "label_ids": torch.tensor(label_ids, dtype=torch.long)
        }