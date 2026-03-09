import torch
import torch.nn as nn
import os
import json
from tqdm import tqdm
from src.utils.logger import get_logger

logger = get_logger("trainer")


class Trainer:

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        device: torch.device,
        checkpoint_dir: str,
        log_every_steps: int = 100,
        save_every_epochs: int = 2
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.log_every_steps = log_every_steps
        self.save_every_epochs = save_every_epochs

        os.makedirs(checkpoint_dir, exist_ok=True)
        self.history = {
            "train_loss": [],
            "val_loss":   []
        }

    def train_epoch(
        self,
        dataloader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        epoch: int
    ) -> float:

        self.model.train()
        total_loss = 0
        total_steps = 0

        progress = tqdm(dataloader, desc=f"Epoch {epoch} train")

        for step, batch in enumerate(progress):
            input_ids = batch["input_ids"].to(self.device)
            labels    = batch["labels"].to(self.device)

            logits, _, _, _ = self.model(input_ids, input_ids)

            logits_flat = logits.reshape(-1, logits.size(-1))
            labels_flat = labels.reshape(-1)

            loss = criterion(logits_flat, labels_flat)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 1.0
            )
            self.optimizer.step()
            
            # Clear MPS cache every 50 steps
            if step % 50 == 0:
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()

            if self.scheduler is not None:
                self.scheduler.step()

            total_loss += loss.item()
            total_steps += 1

            if step % self.log_every_steps == 0:
                avg = total_loss / total_steps
                progress.set_postfix({"loss": f"{avg:.4f}"})
                logger.info(
                    f"Epoch {epoch} | Step {step} | Loss {avg:.4f}"
                )

        return total_loss / total_steps

    def validate_epoch(
        self,
        dataloader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        epoch: int
    ) -> float:

        self.model.eval()
        total_loss = 0
        total_steps = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Epoch {epoch} val"):
                input_ids = batch["input_ids"].to(self.device)
                labels    = batch["labels"].to(self.device)

                logits, _, _, _ = self.model(input_ids, input_ids)

                logits_flat = logits.reshape(-1, logits.size(-1))
                labels_flat = labels.reshape(-1)

                loss = criterion(logits_flat, labels_flat)
                total_loss += loss.item()
                total_steps += 1

        avg_loss = total_loss / total_steps
        logger.info(f"Epoch {epoch} | Val Loss {avg_loss:.4f}")
        return avg_loss

    def save_checkpoint(self, epoch: int, train_loss: float, val_loss: float):
        path = os.path.join(
            self.checkpoint_dir,
            f"checkpoint_epoch_{epoch}.pt"
        )
        torch.save({
            "epoch":      epoch,
            "model":      self.model.state_dict(),
            "optimizer":  self.optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss":   val_loss,
            "history":    self.history
        }, path)
        logger.info(f"Checkpoint saved → {path}")

    def save_history(self):
        path = os.path.join(self.checkpoint_dir, "history.json")
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)

    def run(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader:   torch.utils.data.DataLoader,
        criterion:    nn.Module,
        epochs:       int
    ):
        logger.info(f"Starting training for {epochs} epochs")
        logger.info(f"Device: {self.device}")
        logger.info(f"Checkpoint dir: {self.checkpoint_dir}")

        best_val_loss = float("inf")

        for epoch in range(1, epochs + 1):
            logger.info(f"{'='*50}")
            logger.info(f"EPOCH {epoch}/{epochs}")
            logger.info(f"{'='*50}")

            train_loss = self.train_epoch(
                train_loader, criterion, epoch
            )
            val_loss = self.validate_epoch(
                val_loader, criterion, epoch
            )

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            logger.info(
                f"Epoch {epoch} complete | "
                f"Train: {train_loss:.4f} | "
                f"Val: {val_loss:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, train_loss, val_loss)
                logger.info(f"New best model saved (val_loss={val_loss:.4f})")

            elif epoch % self.save_every_epochs == 0:
                self.save_checkpoint(epoch, train_loss, val_loss)

        self.save_history()
        logger.info("Training complete")
        logger.info(f"Best val loss: {best_val_loss:.4f}")