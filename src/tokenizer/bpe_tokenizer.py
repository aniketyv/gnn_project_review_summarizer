import re
import json
import os
from collections import Counter, defaultdict
from tqdm import tqdm
from src.utils.logger import get_logger

logger = get_logger("tokenizer")


class BPETokenizer:

    def __init__(self, vocab_size: int = 16000):
        self.vocab_size = vocab_size

        # Special tokens — model needs these for specific purposes
        # PAD: fills empty space in batches to make equal length
        # UNK: represents any word not in vocabulary
        # BOS: signals start of a summary to the decoder
        # EOS: signals end of a summary to the decoder
        # SEP: separates multiple reviews from each other
        # MASK: used during pretraining to hide tokens
        self.special_tokens = {
            "<PAD>": 0,
            "<UNK>": 1,
            "<BOS>": 2,
            "<EOS>": 3,
            "<SEP>": 4,
            "<MASK>": 5
        }

        self.token2id = {}
        self.id2token = {}
        self.bpe_merges = {}
        self.vocab = {}
        self.is_trained = False

    def _get_word_freqs(self, texts: list) -> dict:
        # BPE needs to know where words end so it doesnt
        # merge tokens across word boundaries.
        # "the cat" should not merge "e" from "the" with
        # "c" from "cat" into "ec"
        word_freqs = Counter()
        for text in tqdm(texts, desc="Counting words"):
            words = text.lower().split()
            for word in words:
                # Clean word to only letters and apostrophes
                word = re.sub(r"[^a-z']", "", word)
                if len(word) > 1:
                    # Add end of word marker </w>
                    word_freqs[word + "</w>"] += 1
        return dict(word_freqs)

    def _word_to_chars(self, word_freqs: dict) -> dict:
        
        # Converts each word string into a tuple of characters.
        # This is the starting state of BPE — every character
        # is its own token before any merging happens.
        # "amazing</w>" → ("a", "m", "a", "z", "i", "n", "g", "</w>")
        
        vocab = {}
        for word, freq in word_freqs.items():
            chars = tuple(word)
            vocab[chars] = freq
        return vocab

    def _get_pair_freqs(self, vocab: dict) -> dict:
        #  Counts how often each pair of adjacent tokens
        #  appears across all words.

        pair_freqs: dict[tuple[str, str], int] = defaultdict(int)
        for word_tokens, freq in vocab.items():
            for i in range(len(word_tokens) - 1):
                pair = (word_tokens[i], word_tokens[i + 1])
                pair_freqs[pair] += freq
        return dict(pair_freqs)

    def _merge_pair(self, pair: tuple, vocab: dict) -> dict:
        # Merges a specific pair of tokens everywhere
        # it appears in the vocabulary.

        # If pair = ("a", "m"):
        # ("a", "m", "a", "z") → ("am", "a", "z")
        new_vocab = {}
        bigram = pair[0] + pair[1]

        for word_tokens, freq in vocab.items():
            new_tokens = []
            i = 0
            while i < len(word_tokens):
                if (i < len(word_tokens) - 1 and
                        word_tokens[i] == pair[0] and
                        word_tokens[i + 1] == pair[1]):
                    new_tokens.append(bigram)
                    i += 2
                else:
                    new_tokens.append(word_tokens[i])
                    i += 1
            new_vocab[tuple(new_tokens)] = freq

        return new_vocab

    def train(self, texts: list) -> None:
        # Trains the BPE tokenizer on your text data.
        # This is the main training loop.
        
        logger.info(f"Training BPE tokenizer on {len(texts):,} texts")
        logger.info(f"Target vocabulary size: {self.vocab_size:,}")

        # Step 1: Build initial character vocabulary
        word_freqs = self._get_word_freqs(texts)
        logger.info(f"Unique words found: {len(word_freqs):,}")

        self.vocab = self._word_to_chars(word_freqs)

        # Collect all unique characters as base vocabulary
        base_vocab = set()
        for word_tokens in self.vocab.keys():
            for token in word_tokens:
                base_vocab.add(token)

        # Initialize token2id with special tokens first
        self.token2id = dict(self.special_tokens)

        # Add base characters
        for char in sorted(base_vocab):
            if char not in self.token2id:
                self.token2id[char] = len(self.token2id)

        logger.info(f"Base vocabulary size: {len(self.token2id):,}")

        # Step 2: BPE merge loop
        n_merges = self.vocab_size - len(self.token2id)
        logger.info(f"Running {n_merges:,} BPE merges...")

        for merge_idx in tqdm(range(n_merges), desc="BPE merges"):
            # Find most frequent pair
            pair_freqs = self._get_pair_freqs(self.vocab)

            if not pair_freqs:
                logger.info(f"No more pairs to merge at step {merge_idx}")
                break

            best_pair = max(pair_freqs, key=lambda p: pair_freqs[p])
            best_freq = pair_freqs[best_pair]

            # Stop if best pair appears only once — not worth merging
            if best_freq < 2:
                logger.info(f"Best pair frequency too low ({best_freq}), stopping")
                break

            # Merge the best pair everywhere
            self.vocab = self._merge_pair(best_pair, self.vocab)

            # Add merged token to vocabulary
            new_token = best_pair[0] + best_pair[1]
            if new_token not in self.token2id:
                self.token2id[new_token] = len(self.token2id)

            # Remember this merge for encoding later
            self.bpe_merges[best_pair] = len(self.bpe_merges)

        # Build reverse lookup
        self.id2token = {v: k for k, v in self.token2id.items()}
        self.is_trained = True

        logger.info(f"Tokenizer trained. Final vocab size: {len(self.token2id):,}")

    def _tokenize_word(self, word: str) -> list:
        # Applies learned BPE merges to a single word.
        # Used during encoding of new text.
        
        if not word:
            return []

        word = word + "</w>"
        tokens = list(word)

        # Apply merges in the order they were learned
        while len(tokens) > 1:
            pairs = [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]
            # Find the earliest learned merge among current pairs
            mergeable = [(p, self.bpe_merges[p]) for p in pairs if p in self.bpe_merges]
            if not mergeable:
                break
            best_pair = min(mergeable, key=lambda x: x[1])[0]
            new_tokens = []
            i = 0
            while i < len(tokens):
                if (i < len(tokens) - 1 and
                        tokens[i] == best_pair[0] and
                        tokens[i+1] == best_pair[1]):
                    new_tokens.append(best_pair[0] + best_pair[1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return tokens

    def encode(self, text: str, max_length: int = None, # type: ignore
               add_special_tokens: bool = False) -> list:
        # Converts text string into list of integer token ids.
        # This is what gets fed into the model.

        # add_special_tokens=True adds BOS at start and EOS at end
        # Use this when encoding summaries for the decoder.
        # Do not use this when encoding reviews for the encoder.
        if not self.is_trained:
            raise RuntimeError("Tokenizer not trained yet. Call train() first.")

        token_ids = []

        if add_special_tokens:
            token_ids.append(self.special_tokens["<BOS>"])

        words = re.sub(r"[^a-z' ]", "", text.lower()).split()

        for word in words:
            word_tokens = self._tokenize_word(word)
            for token in word_tokens:
                token_id = self.token2id.get(
                    token,
                    self.special_tokens["<UNK>"]
                )
                token_ids.append(token_id)

        if add_special_tokens:
            token_ids.append(self.special_tokens["<EOS>"])

        # Truncate if max_length specified
        if max_length is not None:
            token_ids = token_ids[:max_length]

        return token_ids

    def decode(self, token_ids: list) -> str:
        # Converts list of integer token ids back to text string.
        # Used to read the model's generated output.
        
        tokens = []
        for tid in token_ids:
            token = self.id2token.get(tid, "<UNK>")
            # Skip special tokens when decoding output
            if token in self.special_tokens:
                continue
            tokens.append(token)

        # Join tokens and clean up end-of-word markers
        text = "".join(tokens)
        text = text.replace("</w>", " ")
        text = text.strip()

        return text

    def pad_sequence(self, token_ids: list, max_length: int) -> list:
        # Pads or truncates a sequence to exactly max_length.
        # All sequences in a batch must be the same length
        # for the model to process them together.
        
        pad_id = self.special_tokens["<PAD>"]
        if len(token_ids) >= max_length:
            return token_ids[:max_length]
        return token_ids + [pad_id] * (max_length - len(token_ids))

    def batch_encode(self, texts: list, max_length: int,
                     add_special_tokens: bool = False) -> list:
        # Encodes a list of texts and pads them all
        # to the same length. Returns a list of equal
        # length token id sequences ready for batching.
        
        encoded = []
        for text in texts:
            ids = self.encode(
                text,
                max_length=max_length,
                add_special_tokens=add_special_tokens
            )
            ids = self.pad_sequence(ids, max_length)
            encoded.append(ids)
        return encoded

    def save(self, path: str) -> None:
        # Saves trained tokenizer to disk as JSON.
        # Load once, save forever — never retrain.
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "vocab_size": self.vocab_size,
            "token2id": self.token2id,
            "id2token": {str(k): v for k, v in self.id2token.items()},
            "bpe_merges": {
                f"{k[0]}|||{k[1]}": v
                for k, v in self.bpe_merges.items()
            },
            "special_tokens": self.special_tokens
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Tokenizer saved to {path}")

    def load(self, path: str) -> "BPETokenizer":
        # Loads a previously trained tokenizer from disk.
        # Use this at the start of every training run.
        
        with open(path, "r") as f:
            data = json.load(f)

        self.vocab_size = data["vocab_size"]
        self.token2id = data["token2id"]
        self.id2token = {int(k): v for k, v in data["id2token"].items()}
        self.bpe_merges = {
            tuple(k.split("|||")): v
            for k, v in data["bpe_merges"].items()
        }
        self.special_tokens = data["special_tokens"]
        self.is_trained = True

        logger.info(f"Tokenizer loaded from {path}")
        logger.info(f"Vocabulary size: {len(self.token2id):,}")
        return self