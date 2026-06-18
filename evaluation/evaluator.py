# evaluation/evaluator.py
"""
Detectra Evaluation Engine — Step 6

Loads best_acc.pt checkpoint, runs inference on the full test set,
and computes:
  - Confusion matrix
  - Accuracy, Recall, Specificity, F1-Score, Precision
  - AUC-ROC
  - Saves confusion matrix and ROC curve plots
  - Saves full text report
"""

import logging
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe for background runs
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from pathlib import Path
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    classification_report,
)
from torch.utils.data import DataLoader
from model.Detectra import Detectra

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Loads a saved Detectra checkpoint and evaluates it on the test DataLoader.

    Usage:
        evaluator = Evaluator(checkpoint_path, test_loader, device, results_dir)
        evaluator.run()
    """

    def __init__(
        self,
        checkpoint_path: str,
        test_loader:     DataLoader,
        device:          torch.device,
        results_dir:     Path,
    ):
        self.test_loader    = test_loader
        self.device         = device
        self.results_dir    = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.model = self._load_model(checkpoint_path)

    def _load_model(self, checkpoint_path: str) -> Detectra:
        """
        Load Detectra from a .pt checkpoint.
        Handles float16 → float32 casting automatically.
        """
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {path}\n"
                f"Make sure Step 5 training completed successfully."
            )

        logger.info(f"Loading checkpoint: {path}")
        checkpoint = torch.load(path, map_location="cpu")

        # ── Rebuild model architecture ────────────────────────────────────────
        model = Detectra(pretrained=False)   # pretrained=False — we load our weights

        # ── Cast float16 weights back to float32 if needed ────────────────────
        state_dict = checkpoint["model_state"]
        precision  = checkpoint.get("precision", "float32")

        if precision == "float16":
            logger.info("  Casting float16 checkpoint → float32 for inference")
            state_dict = {
                k: v.float() if v.is_floating_point() else v
                for k, v in state_dict.items()
            }

        model.load_state_dict(state_dict)
        model = model.to(self.device)
        model.eval()   # CRITICAL — disables dropout and batch norm train mode

        epoch    = checkpoint.get("epoch", "?")
        val_acc  = checkpoint.get("val_acc", 0)
        val_loss = checkpoint.get("val_loss", 0)
        logger.info(
            f"  Loaded checkpoint from epoch {epoch} | "
            f"Val acc: {val_acc:.2f}% | Val loss: {val_loss:.4f}"
        )
        return model

    # ── Inference pass ────────────────────────────────────────────────────────

    def _run_inference(self) -> tuple:
        """
        Run model on all test batches.

        Returns:
            all_labels : numpy array of true labels (0=Real, 1=Fake)
            all_preds  : numpy array of predicted labels
            all_probs  : numpy array of P(Fake) scores for AUC-ROC
        """
        all_labels = []
        all_preds  = []
        all_probs  = []

        logger.info(f"Running inference on {len(self.test_loader.dataset)} test videos...")

        with torch.no_grad():
            for batch_idx, (frames, labels) in enumerate(self.test_loader):
                frames = frames.to(self.device)
                labels = labels.to(self.device)

                # return_probs=True → softmax output [P(Real), P(Fake)]
                probs = self.model(frames, return_probs=True)   # (B, 2)
                preds = probs.argmax(dim=1)                     # (B,)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                # Use P(Fake) column for AUC-ROC scoring
                all_probs.extend(probs[:, 1].cpu().numpy())

                if (batch_idx + 1) % 5 == 0:
                    logger.info(
                        f"  Processed {(batch_idx+1)*self.test_loader.batch_size}"
                        f"/{len(self.test_loader.dataset)} videos"
                    )

        return (
            np.array(all_labels),
            np.array(all_preds),
            np.array(all_probs),
        )

    # ── Metric computation ────────────────────────────────────────────────────

    def _compute_metrics(
        self,
        labels: np.ndarray,
        preds:  np.ndarray,
        probs:  np.ndarray,
    ) -> dict:
        """
        Compute all required metrics from predictions.

        Confusion matrix layout (sklearn default, label order [0, 1]):
                        Predicted
                        Real(0)  Fake(1)
        Actual Real(0) [  TN       FP  ]
        Actual Fake(1) [  FN       TP  ]

        Note: TP/FN/FP/TN are from the perspective of FAKE = Positive class.
        """
        # ── Confusion matrix ──────────────────────────────────────────────────
        cm = confusion_matrix(labels, preds, labels=[0, 1])
        # cm[0,0]=TN  cm[0,1]=FP
        # cm[1,0]=FN  cm[1,1]=TP
        TN = int(cm[0, 0])
        FP = int(cm[0, 1])
        FN = int(cm[1, 0])
        TP = int(cm[1, 1])

        total = TP + TN + FP + FN

        # ── Core metrics (as specified in pipeline) ───────────────────────────
        accuracy    = (TP + TN) / total
        recall      = TP / (TP + FN)      if (TP + FN) > 0 else 0.0
        specificity = TN / (TN + FP)      if (TN + FP) > 0 else 0.0
        precision   = TP / (TP + FP)      if (TP + FP) > 0 else 0.0
        f1_score    = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )

        # ── AUC-ROC ───────────────────────────────────────────────────────────
        fpr, tpr, thresholds = roc_curve(labels, probs, pos_label=1)
        auc_score = auc(fpr, tpr)

        return {
            # Confusion matrix components
            "TP": TP, "TN": TN, "FP": FP, "FN": FN,
            # Core metrics
            "accuracy":    accuracy,
            "recall":      recall,        # sensitivity — TP rate on fake videos
            "specificity": specificity,   # TN rate on real videos
            "precision":   precision,
            "f1_score":    f1_score,
            "auc_roc":     auc_score,
            # ROC curve data (for plotting)
            "fpr": fpr, "tpr": tpr, "thresholds": thresholds,
            # Raw
            "confusion_matrix": cm,
            "total_samples": total,
        }

    # ── Confusion matrix plot ─────────────────────────────────────────────────

    def _plot_confusion_matrix(self, cm: np.ndarray, accuracy: float) -> Path:
        """Save a clean confusion matrix heatmap."""
        fig, ax = plt.subplots(figsize=(7, 6))
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#1a1a2e")

        # Colour map: dark blue → teal
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.colorbar(im, ax=ax)

        classes    = ["Real (0)", "Fake (1)"]
        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(classes, fontsize=12, color="white")
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(classes, fontsize=12, color="white")

        # Annotate cells
        cm_max = cm.max()
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                cell_label = {
                    (0, 0): f"TN\n{cm[i,j]}",
                    (0, 1): f"FP\n{cm[i,j]}",
                    (1, 0): f"FN\n{cm[i,j]}",
                    (1, 1): f"TP\n{cm[i,j]}",
                }[(i, j)]
                color = "white" if cm[i, j] < cm_max / 2 else "black"
                ax.text(
                    j, i, cell_label,
                    ha="center", va="center",
                    fontsize=14, fontweight="bold", color=color
                )

        ax.set_xlabel("Predicted Label", fontsize=13, color="white", labelpad=10)
        ax.set_ylabel("True Label",      fontsize=13, color="white", labelpad=10)
        ax.set_title(
            f"Detectra — Confusion Matrix\nAccuracy: {accuracy*100:.2f}%",
            fontsize=14, color="white", pad=15
        )
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("white")

        plt.tight_layout()
        out_path = self.results_dir / "confusion_matrix.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close()
        logger.info(f"  Saved confusion matrix → {out_path}")
        return out_path

    # ── ROC curve plot ────────────────────────────────────────────────────────

    def _plot_roc_curve(self, fpr: np.ndarray, tpr: np.ndarray, auc_score: float) -> Path:
        """Save the AUC-ROC curve plot."""
        fig, ax = plt.subplots(figsize=(7, 6))
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#1a1a2e")

        # ROC curve
        ax.plot(
            fpr, tpr,
            color="#00d4ff", linewidth=2.5,
            label=f"ROC Curve (AUC = {auc_score:.4f})"
        )
        # Random classifier baseline
        ax.plot(
            [0, 1], [0, 1],
            color="#ff6b6b", linewidth=1.5,
            linestyle="--", label="Random Classifier (AUC = 0.50)"
        )
        # Shade area under curve
        ax.fill_between(fpr, tpr, alpha=0.15, color="#00d4ff")

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate (1 - Specificity)",
                      fontsize=12, color="white", labelpad=10)
        ax.set_ylabel("True Positive Rate (Recall/Sensitivity)",
                      fontsize=12, color="white", labelpad=10)
        ax.set_title(
            "Detectra — AUC-ROC Curve",
            fontsize=14, color="white", pad=15
        )
        ax.tick_params(colors="white")
        ax.legend(loc="lower right", fontsize=11,
                  facecolor="#2a2a4e", labelcolor="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444466")

        ax.grid(True, alpha=0.2, color="white")

        plt.tight_layout()
        out_path = self.results_dir / "roc_curve.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close()
        logger.info(f"  Saved ROC curve → {out_path}")
        return out_path

    # ── Text report ───────────────────────────────────────────────────────────

    def _save_report(self, metrics: dict, labels: np.ndarray, preds: np.ndarray) -> Path:
        """Save full evaluation report as a .txt file."""
        report_path = self.results_dir / "evaluation_report.txt"

        sklearn_report = classification_report(
            labels, preds,
            target_names=["Real", "Fake"],
            digits=4
        )

        lines = [
            "=" * 60,
            "  Detectra — EVALUATION REPORT",
            "  Step 6: Test Set Evaluation",
            "=" * 60,
            "",
            f"  Total test samples : {metrics['total_samples']}",
            f"  Real videos        : {metrics['TN'] + metrics['FP']}",
            f"  Fake videos        : {metrics['TP'] + metrics['FN']}",
            "",
            "─" * 60,
            "  CONFUSION MATRIX",
            "─" * 60,
            "",
            "                 Predicted",
            "                 Real    Fake",
            f"  Actual Real  [  {metrics['TN']:<5}   {metrics['FP']:<5}]   (TN, FP)",
            f"  Actual Fake  [  {metrics['FN']:<5}   {metrics['TP']:<5}]   (FN, TP)",
            "",
            "─" * 60,
            "  METRICS (Fake = Positive class)",
            "─" * 60,
            "",
            f"  Accuracy     = (TP+TN)/(TP+TN+FP+FN)",
            f"               = ({metrics['TP']}+{metrics['TN']})/({metrics['total_samples']})",
            f"               = {metrics['accuracy']*100:.4f}%",
            "",
            f"  Recall       = TP/(TP+FN)  [Sensitivity]",
            f"               = {metrics['TP']}/({metrics['TP']}+{metrics['FN']})",
            f"               = {metrics['recall']*100:.4f}%",
            "",
            f"  Specificity  = TN/(TN+FP)",
            f"               = {metrics['TN']}/({metrics['TN']}+{metrics['FP']})",
            f"               = {metrics['specificity']*100:.4f}%",
            "",
            f"  Precision    = TP/(TP+FP)",
            f"               = {metrics['TP']}/({metrics['TP']}+{metrics['FP']})",
            f"               = {metrics['precision']*100:.4f}%",
            "",
            f"  F1-Score     = 2*(Precision*Recall)/(Precision+Recall)",
            f"               = {metrics['f1_score']*100:.4f}%",
            "",
            f"  AUC-ROC      = {metrics['auc_roc']:.4f}",
            "",
            "─" * 60,
            "  SKLEARN CLASSIFICATION REPORT",
            "─" * 60,
            "",
            sklearn_report,
            "=" * 60,
        ]

        with open(report_path, "w") as f:
            f.write("\n".join(lines))

        logger.info(f"  Saved evaluation report → {report_path}")
        return report_path

    # ── Main entry point ──────────────────────────────────────────────────────

    def run(self) -> dict:
        """Run full evaluation pipeline. Returns metrics dict."""

        logger.info("=" * 60)
        logger.info("STEP 6 — EVALUATION")
        logger.info("=" * 60)

        # 1. Run inference
        labels, preds, probs = self._run_inference()

        # 2. Compute all metrics
        logger.info("Computing metrics...")
        metrics = self._compute_metrics(labels, preds, probs)

        # 3. Print metrics to console
        self._print_metrics(metrics)

        # 4. Save plots
        logger.info("Saving plots...")
        self._plot_confusion_matrix(metrics["confusion_matrix"], metrics["accuracy"])
        self._plot_roc_curve(metrics["fpr"], metrics["tpr"], metrics["auc_roc"])

        # 5. Save text report
        self._save_report(metrics, labels, preds)

        logger.info("=" * 60)
        logger.info("EVALUATION COMPLETE")
        logger.info(f"  Results saved to: {self.results_dir}/")
        logger.info("=" * 60)

        return metrics

    def _print_metrics(self, m: dict) -> None:
        """Print a clean metrics summary to console and log."""
        lines = [
            "",
            "=" * 60,
            "  EVALUATION RESULTS",
            "=" * 60,
            f"  Total test samples : {m['total_samples']}",
            "",
            "  CONFUSION MATRIX",
            "                 Predicted",
            "                 Real    Fake",
            f"  Actual Real  [  {m['TN']:<5}   {m['FP']:<5}]",
            f"  Actual Fake  [  {m['FN']:<5}   {m['TP']:<5}]",
            "",
            "  METRICS",
            f"  Accuracy    : {m['accuracy']*100:.2f}%",
            f"  Recall      : {m['recall']*100:.2f}%",
            f"  Specificity : {m['specificity']*100:.2f}%",
            f"  Precision   : {m['precision']*100:.2f}%",
            f"  F1-Score    : {m['f1_score']*100:.2f}%",
            f"  AUC-ROC     : {m['auc_roc']:.4f}",
            "=" * 60,
        ]
        for line in lines:
            logger.info(line)