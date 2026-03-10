import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.model.transformer import Transformer
from src.tokenizer.bpe_tokenizer import BPETokenizer
from src.data.finetune_dataset import FinetuneDataset
from src.training.trainer import Trainer
from src.utils.logger import get_logger
from src.utils.device import get_device
from src.utils.seed import set_seed

logger = get_logger("finetune")


def load_pretrained_model(config, device, checkpoint_path):
    """
    Loads the pretrained transformer weights.
    This is the key step — we start finetuning
    from what the model already learned, not scratch.
    """
    model = Transformer(config).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])

    logger.info(f"Loaded pretrained weights from {checkpoint_path}")
    logger.info(
        f"Pretrain val loss was: "
        f"{checkpoint.get('val_loss', 'unknown'):.4f}"
    )

    return model


def run_finetune(config: dict, device: torch.device):
    set_seed(config["project"]["seed"])

    logger.info("Loading tokenizer...")
    tokenizer = BPETokenizer()
    tokenizer.load("data/processed/tokenizer.json")

    logger.info("Building finetune datasets...")
    train_dataset = FinetuneDataset(
        csv_path="data/processed/train.csv",
        tokenizer=tokenizer,
        src_max_length=config["model"]["max_seq_length"],
        tgt_max_length=config["finetune"]["tgt_max_length"],
        min_reviews_per_business=config["finetune"]["min_reviews_per_business"]
    )
    val_dataset = FinetuneDataset(
        csv_path="data/processed/val.csv",
        tokenizer=tokenizer,
        src_max_length=config["model"]["max_seq_length"],
        tgt_max_length=config["finetune"]["tgt_max_length"],
        min_reviews_per_business=config["finetune"]["min_reviews_per_business"]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["finetune"]["batch_size"],
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["finetune"]["batch_size"],
        shuffle=False,
        num_workers=0
    )

    logger.info("Loading pretrained model...")
    checkpoint_path = config["finetune"]["pretrain_checkpoint"]
    model = load_pretrained_model(config, device, checkpoint_path)

    # Lower learning rate for finetuning
    # We don't want to destroy pretrained weights
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["finetune"]["learning_rate"],
        weight_decay=config["finetune"]["weight_decay"]
    )

    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    trainer = FineTuneTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        checkpoint_dir=config["finetune"]["checkpoint_dir"],
        tokenizer=tokenizer,
        log_every_steps=config["finetune"]["log_every_steps"],
        save_every_epochs=config["finetune"]["save_every_epochs"]
    )

    trainer.run(
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        epochs=config["finetune"]["epochs"]
    )


class FineTuneTrainer(Trainer):
    """
    Extends base Trainer with finetuning specific logic.
    Key difference: uses src_ids and tgt_ids separately
    instead of feeding same input to both encoder and decoder.
    Also generates sample summaries after each epoch
    so we can see quality improving.
    """

    def __init__(self, tokenizer, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer

    def train_epoch(self, dataloader, criterion, epoch):
        self.model.train()
        total_loss = 0
        total_steps = 0

        from tqdm import tqdm
        progress = tqdm(dataloader, desc=f"Epoch {epoch} finetune")

        for step, batch in enumerate(progress):
            src_ids   = batch["src_ids"].to(self.device)
            tgt_ids   = batch["tgt_ids"].to(self.device)
            label_ids = batch["label_ids"].to(self.device)

            logits, _, _, _ = self.model(src_ids, tgt_ids)

            logits_flat = logits.reshape(-1, logits.size(-1))
            labels_flat = label_ids.reshape(-1)

            loss = criterion(logits_flat, labels_flat)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 1.0
            )
            self.optimizer.step()

            if step % 50 == 0:
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()

            total_loss += loss.item()
            total_steps += 1

            if step % self.log_every_steps == 0:
                avg = total_loss / total_steps
                progress.set_postfix({"loss": f"{avg:.4f}"})

        return total_loss / total_steps

    def validate_epoch(self, dataloader, criterion, epoch):
        self.model.eval()
        total_loss = 0
        total_steps = 0

        from tqdm import tqdm

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Epoch {epoch} val"):
                src_ids   = batch["src_ids"].to(self.device)
                tgt_ids   = batch["tgt_ids"].to(self.device)
                label_ids = batch["label_ids"].to(self.device)

                logits, _, _, _ = self.model(src_ids, tgt_ids)

                logits_flat = logits.reshape(-1, logits.size(-1))
                labels_flat = label_ids.reshape(-1)

                loss = criterion(logits_flat, labels_flat)
                total_loss += loss.item()
                total_steps += 1

        avg_loss = total_loss / total_steps

        # Generate a sample summary after each epoch
        # So we can see quality improving with our own eyes
        self._generate_sample(dataloader, epoch)

        return avg_loss

    def _generate_sample(self, dataloader, epoch):
        """Generates one sample summary and logs it."""
        self.model.eval()
        batch = next(iter(dataloader))
        src_ids = batch["src_ids"][:1].to(self.device)

        with torch.no_grad():
            summaries = self.model.generate(
                src_ids,
                self.tokenizer,
                max_length=64,
                temperature=0.7
            ) # type: ignore

        from src.utils.logger import get_logger
        logger = get_logger("finetune")
        logger.info(f"Epoch {epoch} sample summary:")
        logger.info(f"  → '{summaries[0]}'")