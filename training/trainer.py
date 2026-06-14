# training/trainer.py
"""
Detectra Training Engine — Step 5

Handles:
  - One full training epoch (forward, backward, optimizer step)
  - One full evaluation epoch (no gradients)
  - Metric tracking: loss, accuracy, per-epoch history
  - Checkpoint saving: best validation loss + best validation accuracy
  - Epoch-level console + file logging
"""

import time
import logging
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim import Adam
from model.detectra import Detectra

logger = logging.getLogger(__name__)


class Trainer:
    """
    Encapsulates the full training lifecycle for Detectra.

    Usage:
        trainer = Trainer(model, train_loader, test_loader, config)
        trainer.run()
    """

    def __init__(
        self,
        model:           Detectra,
        train_loader:    DataLoader,
        test_loader:     DataLoader,
        config:          dict,
        device:          torch.device,
        checkpoint_dir:  Path,
    ):
        self.model          = model.to(device)
        self.train_loader   = train_loader
        self.test_loader    = test_loader
        self.config         = config
        self.device         = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # ── Loss function ─────────────────────────────────────────────────────
        # CrossEntropyLoss includes log-softmax internally.
        # Do NOT apply softmax to model output during training.
        self.criterion = nn.CrossEntropyLoss()

        # ── Optimizer: Adam with lr=1e-5, weight_decay=1e-5 ──────────────────
        self.optimizer = Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )

        # ── Learning rate scheduler (optional but recommended) ────────────────
        # Reduces LR by 0.5 if val loss plateaus for 3 epochs
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=3,
           
        )

        # ── Tracking ──────────────────────────────────────────────────────────
        self.best_val_loss = float("inf")
        self.best_val_acc  = 0.0
        self.history = {
            "train_loss": [], "train_acc": [],
            "val_loss":   [], "val_acc":   [],
        }

    # ── Single training epoch ─────────────────────────────────────────────────

    def _train_epoch(self, epoch: int) -> tuple:
        """
        Run one full pass over the training DataLoader.
        Returns: (avg_loss, accuracy)
        """
        self.model.train()

        running_loss    = 0.0
        correct         = 0
        total           = 0
        accum_steps     = self.config.get("grad_accum_steps", 1)

        self.optimizer.zero_grad()

        for batch_idx, (frames, labels) in enumerate(self.train_loader):
            frames = frames.to(self.device)   # (B, seq, 3, 112, 112)
            labels = labels.to(self.device)   # (B,) — 0=Real, 1=Fake

            # ── Forward pass ──────────────────────────────────────────────────
            # return_probs=False → raw logits for CrossEntropyLoss
            logits = self.model(frames, return_probs=False)   # (B, 2)
            loss   = self.criterion(logits, labels)

            # ── Scale loss for gradient accumulation ──────────────────────────
            loss = loss / accum_steps
            loss.backward()

            # ── Optimizer step every accum_steps batches ──────────────────────
            if (batch_idx + 1) % accum_steps == 0 or \
               (batch_idx + 1) == len(self.train_loader):
                # Clip gradients to prevent exploding gradients in LSTM
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=1.0
                )
                self.optimizer.step()
                self.optimizer.zero_grad()

            # ── Metrics ───────────────────────────────────────────────────────
            running_loss += loss.item() * accum_steps   # unscale for logging
            preds         = logits.argmax(dim=1)
            correct      += (preds == labels).sum().item()
            total        += labels.size(0)

            # ── Batch-level progress log (every 10 batches) ───────────────────
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(self.train_loader):
                batch_acc = 100.0 * correct / total
                logger.info(
                    f"  Epoch {epoch} | Batch {batch_idx+1:>4}/{len(self.train_loader)} "
                    f"| Loss: {running_loss/(batch_idx+1):.4f} "
                    f"| Acc: {batch_acc:.2f}%"
                )

        avg_loss = running_loss / len(self.train_loader)
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy

    # ── Single evaluation epoch ───────────────────────────────────────────────

    def _eval_epoch(self) -> tuple:
        """
        Run one full pass over the test DataLoader with no gradients.
        Returns: (avg_loss, accuracy)
        """
        self.model.eval()

        running_loss = 0.0
        correct      = 0
        total        = 0

        with torch.no_grad():
            for frames, labels in self.test_loader:
                frames = frames.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(frames, return_probs=False)
                loss   = self.criterion(logits, labels)

                running_loss += loss.item()
                preds         = logits.argmax(dim=1)
                correct      += (preds == labels).sum().item()
                total        += labels.size(0)

        avg_loss = running_loss / len(self.test_loader)
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy

    # ── Checkpoint saving ─────────────────────────────────────────────────────

    def _save_checkpoint(self, epoch: int, val_loss: float, val_acc: float, tag: str) -> None:
        """
        Save model + optimizer + training state to a .pt file.

        Args:
            tag : "best_loss" or "best_acc" — determines filename
        """
        model_state_fp16 = {
        k: v.half() if v.is_floating_point() else v
        for k, v in self.model.state_dict().items()
    }
        checkpoint = {
            "epoch":          epoch,
            "model_state":    model_state_fp16,
            "optimizer_state": self.optimizer.state_dict(),
            "val_loss":       val_loss,
            "val_acc":        val_acc,
            "config":         self.config,
            "history":        self.history,
            "precision":       "float16",
        }
        path = self.checkpoint_dir / f"{tag}.pt"
        torch.save(checkpoint, path)
        logger.info(f"  ✓ Checkpoint saved → {path}  "
                    f"(epoch={epoch}, val_loss={val_loss:.4f}, val_acc={val_acc:.2f}%)")

    # ── Main training loop ────────────────────────────────────────────────────

    def run(self) -> dict:
        """
        Run the full training loop for config['num_epochs'] epochs.
        Returns the complete history dict.
        """
        num_epochs = self.config["num_epochs"]

        logger.info("=" * 65)
        logger.info("STEP 5 — TRAINING STARTED")
        logger.info(f"  Epochs        : {num_epochs}")
        logger.info(f"  Learning rate : {self.config['learning_rate']}")
        logger.info(f"  Weight decay  : {self.config['weight_decay']}")
        logger.info(f"  Batch size    : {self.config['batch_size']}")
        logger.info(f"  Device        : {self.device}")
        logger.info(f"  Grad accum    : {self.config.get('grad_accum_steps', 1)} steps")
        logger.info("=" * 65)

        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()

            logger.info(f"\n{'─'*65}")
            logger.info(f"  EPOCH {epoch}/{num_epochs}")
            logger.info(f"{'─'*65}")

            # ── Training phase ────────────────────────────────────────────────
            logger.info("  [TRAIN]")
            train_loss, train_acc = self._train_epoch(epoch)

            # ── Evaluation phase ──────────────────────────────────────────────
            logger.info("  [EVAL]")
            val_loss, val_acc = self._eval_epoch()

            # ── LR scheduler step ─────────────────────────────────────────────
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]["lr"]
            logger.info(f"  Current LR: {current_lr:.2e}")

            # ── History ───────────────────────────────────────────────────────
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            # ── Epoch summary ─────────────────────────────────────────────────
            elapsed = time.time() - epoch_start
            logger.info(
                f"\n  Epoch {epoch:>3} Summary | "
                f"Time: {elapsed:.1f}s | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
            )

            # ── Save best checkpoints ─────────────────────────────────────────
            improved = []

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_checkpoint(epoch, val_loss, val_acc, tag="best_loss")
                improved.append(f"val_loss → {val_loss:.4f}")

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self._save_checkpoint(epoch, val_loss, val_acc, tag="best_acc")
                improved.append(f"val_acc → {val_acc:.2f}%")

            if improved:
                logger.info(f"  🏆 New best: {' | '.join(improved)}")
            else:
                logger.info(
                    f"  No improvement. "
                    f"Best so far → loss={self.best_val_loss:.4f}, "
                    f"acc={self.best_val_acc:.2f}%"
                )

        # ── Training complete ─────────────────────────────────────────────────
        logger.info("\n" + "=" * 65)
        logger.info("TRAINING COMPLETE")
        logger.info(f"  Best val loss : {self.best_val_loss:.4f}  → checkpoints/best_loss.pt")
        logger.info(f"  Best val acc  : {self.best_val_acc:.2f}%  → checkpoints/best_acc.pt")
        logger.info("=" * 65)

        return self.history