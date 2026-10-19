from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
HISTORY_PATH = ROOT / "reports" / "training_history.csv"
FIGURES_DIR = ROOT / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def main():
    if not HISTORY_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {HISTORY_PATH}. Place this script in the project root "
            "and make sure reports/training_history.csv exists."
        )

    history = pd.read_csv(HISTORY_PATH)
    required = {"epoch", "train_loss", "val_loss", "val_mean_auc", "learning_rate"}
    missing = required.difference(history.columns)
    if missing:
        raise ValueError(f"Missing required CSV columns: {sorted(missing)}")

    best_row = history.loc[history["val_mean_auc"].idxmax()]
    best_epoch = int(best_row["epoch"])

    # 1. Training and validation loss
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history["epoch"], history["train_loss"], marker="o", label="Training loss")
    ax.plot(history["epoch"], history["val_loss"], marker="o", label="Validation loss")
    ax.axvline(best_epoch, linestyle="--", label=f"Best validation AUC (epoch {best_epoch})")
    ax.set(title="Training and Validation Loss", xlabel="Epoch", ylabel="BCE loss")
    ax.set_xticks(history["epoch"])
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "loss_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 2. Validation macro ROC-AUC
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history["epoch"], history["val_mean_auc"], marker="o", label="Validation macro ROC-AUC")
    ax.scatter([best_epoch], [best_row["val_mean_auc"]], marker="*", s=180,
               label=f"Best: {best_row['val_mean_auc']:.4f} (epoch {best_epoch})")
    ax.set(title="Validation Macro ROC-AUC by Epoch", xlabel="Epoch", ylabel="Macro ROC-AUC")
    ax.set_xticks(history["epoch"])
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "validation_macro_auc.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 3. Per-disease validation ROC-AUC
    auc_columns = [column for column in history.columns if column.startswith("auc_")]
    if auc_columns:
        fig, ax = plt.subplots(figsize=(11, 7))
        for column in auc_columns:
            ax.plot(history["epoch"], history[column], marker=".", label=column.removeprefix("auc_"))
        ax.axvline(best_epoch, linestyle="--", label=f"Best macro AUC (epoch {best_epoch})")
        ax.set(title="Validation ROC-AUC by Disease", xlabel="Epoch", ylabel="ROC-AUC")
        ax.set_xticks(history["epoch"])
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=2, fontsize=8)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "per_disease_validation_auc.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    # 4. Learning rate schedule
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history["epoch"], history["learning_rate"], marker="o")
    ax.set(title="Learning Rate by Epoch", xlabel="Epoch", ylabel="Learning rate")
    ax.set_xticks(history["epoch"])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "learning_rate.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Best epoch by validation macro ROC-AUC: {best_epoch}")
    print(f"Best validation macro ROC-AUC: {best_row['val_mean_auc']:.4f}")
    print(f"Figures saved in: {FIGURES_DIR}")
    for path in sorted(FIGURES_DIR.glob("*.png")):
        print(f" - {path.name}")


if __name__ == "__main__":
    main()
