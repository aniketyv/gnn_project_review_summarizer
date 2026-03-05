# main.py

import yaml
import argparse
from src.utils.logger import get_logger
from src.utils.device import get_device, get_device_info
from src.utils.seed import set_seed

logger = get_logger("main")


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="GNN Review Summarizer"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["pretrain", "finetune", "evaluate", "infer"],
        required=True,
        help="What to run"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config file"
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    logger.info(f"Project: {config['project']['name']}")
    logger.info(f"Version: {config['project']['version']}")

    # Set seed for reproducibility
    set_seed(config["project"]["seed"])

    # Log device info
    device_info = get_device_info()
    logger.info(f"PyTorch version: {device_info['pytorch_version']}")
    logger.info(f"MPS available: {device_info['mps_available']}")
    logger.info(f"CUDA available: {device_info['cuda_available']}")

    # Get device
    device = get_device(config)
    logger.info(f"Using device: {device}")

    # Route to correct module
    if args.mode == "pretrain":
        logger.info("Starting pretraining...")
        from src.training.pretrain import run_pretrain
        run_pretrain(config, device)

    elif args.mode == "finetune":
        logger.info("Starting finetuning...")
        from src.training.finetune import run_finetune
        run_finetune(config, device)

    elif args.mode == "evaluate":
        logger.info("Starting evaluation...")
        from src.evaluation.metrics import run_evaluation
        run_evaluation(config, device)

    elif args.mode == "infer":
        logger.info("Starting inference...")
        # Will implement after training is complete


if __name__ == "__main__":
    main()