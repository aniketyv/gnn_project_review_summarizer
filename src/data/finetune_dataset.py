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
        """
        FIX 2: Use TextRank to extract the most representative
        sentence from a review instead of just taking the first
        sentence.

        TextRank ranks sentences by their similarity to all other
        sentences in the review. The highest ranked sentence is
        the most central opinion — not just the first thing
        the reviewer wrote.

        Falls back to first sentence if TextRank fails.
        """
        try:
            from sumy.parsers.plaintext import PlaintextParser
            from sumy.nlp.tokenizers import Tokenizer
            from sumy.summarizers.text_rank import TextRankSummarizer

            parser = PlaintextParser.from_string(
                text, Tokenizer("english")
            )
            summarizer = TextRankSummarizer()
            sentences = summarizer(parser.document, sentences_count=1)

            if sentences:
                result = str(sentences[0]).strip().lower()
                if len(result.split()) >= 5:
                    return result[:200]

        except Exception:
            pass

        # Fallback: first meaningful sentence
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

        if not sentences:
            return text[:100].lower()

        summary = sentences[0]
        if len(sentences) > 1 and len(summary) < 60:
            summary = summary + " " + sentences[1]

        return summary[:200].lower()

    def _build_review_summary_pairs(
        self,
        df: pd.DataFrame,
        min_reviews: int
    ) -> list:
        """
        FIX 1: Add average star rating as explicit sentiment signal.
        Prepend [STARS: X.X] to each input so the model knows
        whether to generate a positive or negative summary.

        FIX 2: Each review takes a turn being the summary target,
        and TextRank extracts the most representative sentence
        rather than the first sentence.

        Together these fixes address:
        - Wrong sentiment (Business 4: 1.8 stars got positive summary)
        - Generic phrases (model had no sentiment anchor)
        - Specificity (TextRank picks content-rich sentences)
        """
        pairs = []

        grouped = df.groupby("business_id")

        for business_id, group in grouped:
            if len(group) < min_reviews:
                continue

            group = group.sort_values("useful", ascending=False)
            reviews = group["text"].tolist()

            # Get star ratings aligned with reviews
            stars = group["stars"].tolist() \
                if "stars" in group.columns else [3.0] * len(reviews)

            for i in range(len(reviews)):
                summary = self._extract_summary(reviews[i])

                # Collect other reviews and their star ratings
                other_texts = [
                    reviews[j]
                    for j in range(len(reviews)) if j != i
                ][:5]

                other_stars = [
                    float(stars[j])
                    for j in range(len(reviews)) if j != i
                ][:5]

                if not other_texts:
                    continue

                # FIX 1: Compute average stars and prepend as signal
                # Round to nearest 0.5 for cleaner token
                avg_stars = sum(other_stars) / len(other_stars)
                rounded_stars = round(avg_stars * 2) / 2
                star_prefix = f"[STARS: {rounded_stars}]"

                combined_input = (
                    star_prefix + " " +
                    " [SEP] ".join(other_texts)
                )

                if len(summary.split()) < 5:
                    continue
                if len(combined_input.split()) < 20:
                    continue

                pairs.append({
                    "input":       combined_input,
                    "summary":     summary,
                    "business_id": business_id,
                    "avg_stars":   avg_stars
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
            "src_ids":   torch.tensor(src_ids,   dtype=torch.long),
            "tgt_ids":   torch.tensor(tgt_ids,   dtype=torch.long),
            "label_ids": torch.tensor(label_ids, dtype=torch.long)
        }