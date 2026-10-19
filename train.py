
from pathlib import Path
import csv

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from src.dataset import LABELS, create_dataloaders
from src.model import create_model


# ---------------- Configuration ----------------

ROOT = Path(__file__).resolve().parent

BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 1e-4
MAX_POS_WEIGHT = 10.0

EARLY_STOPPING_PATIENCE = 3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT_DIR = ROOT / "models"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# Keep the original baseline checkpoint untouched.
BEST_MODEL_PATH = CHECKPOINT_DIR / "best_resnet18_extended.pth"

REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
HISTORY_PATH = REPORTS_DIR / "training_history.csv"


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    total_images = 0
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)

            with torch.amp.autocast(
                device_type=DEVICE.type,
                enabled=(DEVICE.type == "cuda"),
            ):
                logits = model(images)
                loss = criterion(logits, targets)

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_images += batch_size

            all_probs.append(torch.sigmoid(logits).cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    probabilities = np.concatenate(all_probs)
    targets = np.concatenate(all_targets)

    per_label_auc = {}

    for i, label in enumerate(LABELS):
        if len(np.unique(targets[:, i])) == 2:
            per_label_auc[label] = roc_auc_score(
                targets[:, i], probabilities[:, i]
            )
        else:
            per_label_auc[label] = float("nan")

    mean_auc = float(np.nanmean(list(per_label_auc.values())))
    average_loss = total_loss / total_images

    return average_loss, mean_auc, per_label_auc


def main():
    print("Device:", DEVICE)

    if DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    loaders = create_dataloaders(
        root_dir=ROOT,
        batch_size=BATCH_SIZE,
        num_workers=0,
    )

    train_dataset = loaders["train"].dataset

    # Calculate positive class weights using training labels only.
    label_counts = train_dataset.labels.sum(dim=0)
    total_train = len(train_dataset)
    negative_counts = total_train - label_counts

    pos_weight = (
        negative_counts / label_counts.clamp(min=1)
    ).clamp(max=MAX_POS_WEIGHT).to(DEVICE)

    print("\nPositive class weights:")
    for label, weight in zip(LABELS, pos_weight.tolist()):
        print(f"  {label}: {weight:.2f}")

    # Start a fresh run from ImageNet-pretrained ResNet-18.
    model = create_model(pretrained=True).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=1,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(DEVICE.type == "cuda"),
    )

    best_auc = -float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    history_fields = (
        ["epoch", "train_loss", "val_loss", "val_mean_auc", "learning_rate"]
        + [f"auc_{label}" for label in LABELS]
    )

    # Start a fresh CSV for this experiment.
    with open(HISTORY_PATH, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=history_fields)
        writer.writeheader()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        total_images = 0

        for step, (images, targets) in enumerate(
            loaders["train"], start=1
        ):
            images = images.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type=DEVICE.type,
                enabled=(DEVICE.type == "cuda"),
            ):
                logits = model(images)
                loss = criterion(logits, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            total_images += batch_size

            if step % 500 == 0:
                print(
                    f"Epoch {epoch}/{EPOCHS} | "
                    f"Step {step}/{len(loaders['train'])} | "
                    f"Train loss: {running_loss / total_images:.4f}"
                )

        train_loss = running_loss / total_images

        val_loss, val_auc, per_label_auc = evaluate(
            model, loaders["val"], criterion
        )

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_auc)

        print(f"\nEpoch {epoch}/{EPOCHS} complete")
        print(f"Train loss:     {train_loss:.4f}")
        print(f"Validation loss:{val_loss: .4f}")
        print(f"Validation AUC: {val_auc:.4f}")
        print(f"Learning rate:  {current_lr:.2e}")

        for label, auc in per_label_auc.items():
            print(f"  {label}: {auc:.4f}")

        # Log every epoch for learning-curve plots.
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_mean_auc": val_auc,
            "learning_rate": current_lr,
        }

        for label, auc in per_label_auc.items():
            row[f"auc_{label}"] = auc

        with open(HISTORY_PATH, "a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=history_fields)
            writer.writerow(row)

        # Save the best model according to validation ROC-AUC.
        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "labels": LABELS,
                    "val_mean_auc": val_auc,
                    "val_loss": val_loss,
                },
                BEST_MODEL_PATH,
            )

            print(f"Saved best checkpoint: {BEST_MODEL_PATH}")
        else:
            epochs_without_improvement += 1
            print(
                "No validation AUC improvement: "
                f"{epochs_without_improvement}/"
                f"{EARLY_STOPPING_PATIENCE} epochs"
            )

        print("-" * 60)

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(
                f"Early stopping at epoch {epoch}. "
                f"Best epoch was {best_epoch}."
            )
            break

    print("\nTraining complete.")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation ROC-AUC: {best_auc:.4f}")
    print(f"Best checkpoint: {BEST_MODEL_PATH}")
    print(f"Training history: {HISTORY_PATH}")


if __name__ == "__main__":
    main()
