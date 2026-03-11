import torch
import pandas as pd
from tqdm import tqdm
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import json
import os
import sys
sys.path.append(".")

from src.model.transformer import Transformer
from src.tokenizer.bpe_tokenizer import BPETokenizer
from src.data.loader import load_config
from src.data.finetune_dataset import FinetuneDataset
from src.utils.logger import get_logger

logger = get_logger("evaluation")


def load_model(config, checkpoint_path, device):
    model = Transformer(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    logger.info(f"Model loaded from {checkpoint_path}")
    return model


def generate_summary(model, tokenizer, input_text, config, device):
    src_ids = tokenizer.encode(
        input_text,
        max_length=config["model"]["max_seq_length"]
    )
    src_ids = tokenizer.pad_sequence(
        src_ids,
        config["model"]["max_seq_length"]
    )
    src_tensor = torch.tensor([src_ids], dtype=torch.long).to(device)

    with torch.no_grad():
        summaries = model.generate(
            src_tensor,
            tokenizer,
            max_length=config["finetune"]["max_summary_length"],
            temperature=0.7,
            top_p=0.92
        )
    return summaries[0]


def compute_rouge(predictions, references):
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True
    )

    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []

    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        rouge1_scores.append(scores["rouge1"].fmeasure)
        rouge2_scores.append(scores["rouge2"].fmeasure)
        rougeL_scores.append(scores["rougeL"].fmeasure)

    return {
        "rouge1": sum(rouge1_scores) / len(rouge1_scores),
        "rouge2": sum(rouge2_scores) / len(rouge2_scores),
        "rougeL": sum(rougeL_scores) / len(rougeL_scores),
    }


def compute_bert_score(predictions, references):
    logger.info("Computing BERTScore (this takes ~2 min)...")
    P, R, F1 = bert_score(
        predictions,
        references,
        lang="en",
        verbose=False
    )
    return {
        "bertscore_precision": P.mean().item(), # type: ignore
        "bertscore_recall":    R.mean().item(), # type: ignore
        "bertscore_f1":        F1.mean().item(), # type: ignore
    }


def run_evaluation(n_samples=200):
    config   = load_config()
    device   = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    logger.info(f"Device: {device}")

    tokenizer = BPETokenizer()
    tokenizer.load("data/processed/tokenizer.json")

    checkpoint = "checkpoints/finetune/checkpoint_epoch_30.pt"
    model = load_model(config, checkpoint, device)

    # Build dataset from test split
    logger.info("Building evaluation dataset from test.csv...")
    dataset = FinetuneDataset(
        csv_path="data/processed/test.csv",
        tokenizer=tokenizer,
        src_max_length=config["model"]["max_seq_length"],
        tgt_max_length=config["finetune"]["tgt_max_length"],
        min_reviews_per_business=config["finetune"]["min_reviews_per_business"]
    )

    logger.info(f"Test pairs available: {len(dataset)}")
    n_samples = min(n_samples, len(dataset))
    logger.info(f"Evaluating on {n_samples} samples...")

    predictions = []
    references  = []

    for i in tqdm(range(n_samples), desc="Generating summaries"):
        sample = dataset.pairs[i]

        generated = generate_summary(
            model,
            tokenizer,
            sample["input"],
            config,
            device
        )

        predictions.append(generated)
        references.append(sample["summary"])

    # Compute metrics
    logger.info("Computing ROUGE scores...")
    rouge_scores = compute_rouge(predictions, references)

    logger.info("Computing BERTScore...")
    bert_scores = compute_bert_score(predictions, references)

    # Combine results
    results = {**rouge_scores, **bert_scores, "n_samples": n_samples}

    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print(f"Model: GNN Opinion Summarizer (trained from scratch)")
    print(f"Samples evaluated: {n_samples}")
    print("="*50)
    print(f"ROUGE-1:           {rouge_scores['rouge1']:.4f}")
    print(f"ROUGE-2:           {rouge_scores['rouge2']:.4f}")
    print(f"ROUGE-L:           {rouge_scores['rougeL']:.4f}")
    print(f"BERTScore F1:      {bert_scores['bertscore_f1']:.4f}")
    print("="*50)

    # Show 5 example predictions vs references
    print("\nSAMPLE PREDICTIONS vs REFERENCES")
    print("-"*50)
    for i in range(15):
        print(f"\nSample {i+1}:")
        print(f"  Reference:  {references[i]}")
        print(f"  Generated:  {predictions[i]}")

    # Save to file
    os.makedirs("results/scores", exist_ok=True)
    with open("results/scores/evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to results/scores/evaluation_results.json")

    return results


if __name__ == "__main__":
    run_evaluation(n_samples=2000)