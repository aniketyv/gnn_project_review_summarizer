import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.model.transformer import Transformer
from src.tokenizer.bpe_tokenizer import BPETokenizer
from src.data.pretrain_dataset import PretrainDataset
from src.training.trainer import Trainer
from src.utils.logger import get_logger
from src.utils.device import get_device
from src.utils.seed import set_seed

logger = get_logger("pretrain")


def get_linear_warmup_scheduler(optimizer, warmup_steps: int):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return 1.0
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def run_pretrain(config: dict, device: torch.device):
    set_seed(config["project"]["seed"])

    logger.info("Loading tokenizer...")
    tokenizer = BPETokenizer()
    tokenizer.load("data/processed/tokenizer.json")

    logger.info("Building datasets...")
    train_dataset = PretrainDataset(
        csv_path="data/processed/train.csv",
        tokenizer=tokenizer,
        max_seq_length=config["model"]["max_seq_length"],
        mask_probability=config["pretrain"]["mask_probability"]
    )
    val_dataset = PretrainDataset(
        csv_path="data/processed/val.csv",
        tokenizer=tokenizer,
        max_seq_length=config["model"]["max_seq_length"],
        mask_probability=config["pretrain"]["mask_probability"]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["pretrain"]["batch_size"],
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["pretrain"]["batch_size"],
        shuffle=False,
        num_workers=0
    )

    logger.info("Building model...")
    model = Transformer(config).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["pretrain"]["learning_rate"],
        weight_decay=config["pretrain"]["weight_decay"]
    )

    total_steps = len(train_loader) * config["pretrain"]["epochs"]
    scheduler = get_linear_warmup_scheduler(
        optimizer,
        warmup_steps=config["pretrain"]["warmup_steps"]
    )

    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=config["pretrain"]["checkpoint_dir"],
        log_every_steps=config["pretrain"]["log_every_steps"],
        save_every_epochs=config["pretrain"]["save_every_epochs"]
    )

    trainer.run(
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        epochs=config["pretrain"]["epochs"]
    )