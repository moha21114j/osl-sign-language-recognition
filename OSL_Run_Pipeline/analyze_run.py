import re
import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze one finished training run.")
    parser.add_argument("--run_dir", type=str, required=True,
                        help="Path to experiment folder containing train.log")
    parser.add_argument("--pred_csv", type=str, required=True,
                        help="CSV file with columns: video_id,true_label,pred_label")
    parser.add_argument("--label_map", type=str, default=None,
                        help="Optional JSON file mapping numeric labels to word names")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Optional output folder. Defaults to run_dir/analysis")
    parser.add_argument("--top_k", type=int, default=15,
                        help="How many best/worst words to show")
    return parser.parse_args()


def load_label_map(label_map_path):
    if label_map_path is None:
        return None
    with open(label_map_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Normalize keys if numeric labels are stored as strings in JSON
    normalized = {}
    for k, v in data.items():
        try:
            normalized[int(k)] = v
        except Exception:
            normalized[k] = v
    return normalized


def apply_label_map(df, label_map):
    if label_map is None:
        return df

    def map_value(x):
        try:
            xi = int(x)
            return label_map.get(xi, x)
        except Exception:
            return label_map.get(x, x)

    df["true_label_name"] = df["true_label"].apply(map_value)
    df["pred_label_name"] = df["pred_label"].apply(map_value)
    return df


def parse_train_log(log_path):
    """
    Tries to extract:
    - epoch
    - lr
    - train loss
    - val loss
    - accuracy
    from free-form train.log
    """

    if not log_path.exists():
        return pd.DataFrame()

    epoch_rows = []

    # Very flexible regex patterns
    epoch_patterns = [
        re.compile(r"epoch\s*[:=\[]?\s*(\d+)", re.IGNORECASE),
        re.compile(r"Epoch\s*\[(\d+)", re.IGNORECASE),
    ]
    lr_patterns = [
        re.compile(r"\blr\b\s*[:=]\s*([0-9.eE+-]+)"),
        re.compile(r"learning[_ ]rate\s*[:=]\s*([0-9.eE+-]+)", re.IGNORECASE),
    ]
    train_loss_patterns = [
        re.compile(r"train[_ ]?loss\s*[:=]\s*([0-9.eE+-]+)", re.IGNORECASE),
        re.compile(r"loss\s*[:=]\s*([0-9.eE+-]+)", re.IGNORECASE),
    ]
    val_loss_patterns = [
        re.compile(r"val[_ ]?loss\s*[:=]\s*([0-9.eE+-]+)", re.IGNORECASE),
        re.compile(r"valid[_ ]?loss\s*[:=]\s*([0-9.eE+-]+)", re.IGNORECASE),
    ]
    acc_patterns = [
        re.compile(r"\bacc(?:uracy)?\b\s*[:=]\s*([0-9.eE+-]+)", re.IGNORECASE),
        re.compile(r"top1\s*[:=]\s*([0-9.eE+-]+)", re.IGNORECASE),
        re.compile(r"val[_ ]?acc(?:uracy)?\s*[:=]\s*([0-9.eE+-]+)", re.IGNORECASE),
    ]

    current = defaultdict(lambda: None)

    def extract_first(patterns, line):
        for pat in patterns:
            m = pat.search(line)
            if m:
                try:
                    return float(m.group(1))
                except Exception:
                    return m.group(1)
        return None

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            epoch_num = None
            for pat in epoch_patterns:
                m = pat.search(line)
                if m:
                    try:
                        epoch_num = int(m.group(1))
                    except Exception:
                        pass
                    break

            lr = extract_first(lr_patterns, line)
            train_loss = extract_first(train_loss_patterns, line)
            val_loss = extract_first(val_loss_patterns, line)
            acc = extract_first(acc_patterns, line)

            # If this line starts a new epoch, save previous row if meaningful
            if epoch_num is not None:
                if current.get("epoch") is not None:
                    epoch_rows.append(dict(current))
                    current = defaultdict(lambda: None)
                current["epoch"] = epoch_num

            if lr is not None:
                current["lr"] = lr
            if train_loss is not None:
                current["train_loss"] = train_loss
            if val_loss is not None:
                current["val_loss"] = val_loss
            if acc is not None:
                current["accuracy"] = acc

    if current.get("epoch") is not None:
        epoch_rows.append(dict(current))

    if not epoch_rows:
        return pd.DataFrame()

    df = pd.DataFrame(epoch_rows).drop_duplicates(subset=["epoch"], keep="last")
    df = df.sort_values("epoch").reset_index(drop=True)
    return df


def compute_per_word_stats(df):
    rows = []
    grouped = df.groupby("true_label_name")

    for label, g in grouped:
        total = len(g)
        correct = int((g["true_label_name"] == g["pred_label_name"]).sum())
        wrong = total - correct
        acc = correct / total if total > 0 else 0.0

        wrong_preds = (
            g[g["true_label_name"] != g["pred_label_name"]]["pred_label_name"]
            .value_counts()
            .head(5)
            .to_dict()
        )

        rows.append({
            "word": label,
            "total_samples": total,
            "correct": correct,
            "wrong": wrong,
            "accuracy": acc,
            "top_confusions": wrong_preds,
        })

    stats_df = pd.DataFrame(rows).sort_values(
        by=["accuracy", "total_samples"], ascending=[False, False]
    ).reset_index(drop=True)

    return stats_df


def save_confusion_matrix(df, out_dir):
    labels = sorted(df["true_label_name"].astype(str).unique().tolist())
    cm = confusion_matrix(df["true_label_name"], df["pred_label_name"], labels=labels)

    fig_size = max(12, min(30, int(len(labels) * 0.35)))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, xticks_rotation=90, colorbar=False)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    out_path = out_dir / "confusion_matrix.png"
    plt.savefig(out_path, dpi=220)
    plt.close()
    return out_path


def save_training_curves(log_df, out_dir):
    saved = []

    if log_df.empty:
        return saved

    if "train_loss" in log_df.columns and log_df["train_loss"].notna().any():
        plt.figure(figsize=(8, 5))
        plt.plot(log_df["epoch"], log_df["train_loss"], marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Train Loss")
        plt.title("Train Loss vs Epoch")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        p = out_dir / "train_loss_curve.png"
        plt.savefig(p, dpi=180)
        plt.close()
        saved.append(p)

    if "val_loss" in log_df.columns and log_df["val_loss"].notna().any():
        plt.figure(figsize=(8, 5))
        plt.plot(log_df["epoch"], log_df["val_loss"], marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Validation Loss")
        plt.title("Validation Loss vs Epoch")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        p = out_dir / "val_loss_curve.png"
        plt.savefig(p, dpi=180)
        plt.close()
        saved.append(p)

    if "accuracy" in log_df.columns and log_df["accuracy"].notna().any():
        plt.figure(figsize=(8, 5))
        plt.plot(log_df["epoch"], log_df["accuracy"], marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Accuracy vs Epoch")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        p = out_dir / "accuracy_curve.png"
        plt.savefig(p, dpi=180)
        plt.close()
        saved.append(p)

    if "lr" in log_df.columns and log_df["lr"].notna().any():
        plt.figure(figsize=(8, 5))
        plt.plot(log_df["epoch"], log_df["lr"], marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.title("Learning Rate vs Epoch")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        p = out_dir / "lr_curve.png"
        plt.savefig(p, dpi=180)
        plt.close()
        saved.append(p)

    return saved


def main():
    args = parse_args()

    run_dir = Path(args.run_dir)
    pred_csv = Path(args.pred_csv)
    out_dir = Path(args.out_dir) if args.out_dir else (run_dir / "analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = run_dir / "train.log"
    label_map = load_label_map(args.label_map)

    # -----------------------------
    # Load predictions
    # -----------------------------
    df = pd.read_csv(pred_csv)

    needed_cols = {"true_label", "pred_label"}
    if not needed_cols.issubset(df.columns):
        raise ValueError(f"Prediction CSV must contain columns: {needed_cols}")

    if "video_id" not in df.columns:
        df["video_id"] = np.arange(len(df))

    df = apply_label_map(df, label_map)

    if "true_label_name" not in df.columns:
        df["true_label_name"] = df["true_label"].astype(str)
    if "pred_label_name" not in df.columns:
        df["pred_label_name"] = df["pred_label"].astype(str)

    # -----------------------------
    # Overall metrics
    # -----------------------------
    overall_acc = accuracy_score(df["true_label_name"], df["pred_label_name"])
    report = classification_report(
        df["true_label_name"],
        df["pred_label_name"],
        output_dict=True,
        zero_division=0
    )
    report_df = pd.DataFrame(report).transpose()

    # -----------------------------
    # Per-word metrics
    # -----------------------------
    per_word_df = compute_per_word_stats(df)

    good_words = per_word_df.sort_values(
        by=["accuracy", "total_samples"], ascending=[False, False]
    ).head(args.top_k)

    bad_words = per_word_df.sort_values(
        by=["accuracy", "total_samples"], ascending=[True, False]
    ).head(args.top_k)

    # -----------------------------
    # Parse training log
    # -----------------------------
    log_df = parse_train_log(log_path)

    # Best epoch by accuracy if available
    best_epoch_info = {}
    if not log_df.empty and "accuracy" in log_df.columns and log_df["accuracy"].notna().any():
        best_row = log_df.loc[log_df["accuracy"].astype(float).idxmax()]
        best_epoch_info = {
            "best_epoch_by_accuracy": int(best_row["epoch"]),
            "best_accuracy": float(best_row["accuracy"]),
        }

    # -----------------------------
    # Save outputs
    # -----------------------------
    df.to_csv(out_dir / "predictions_with_names.csv", index=False)
    report_df.to_csv(out_dir / "classification_report.csv")
    per_word_df.to_csv(out_dir / "per_word_stats.csv", index=False)
    good_words.to_csv(out_dir / "good_words.csv", index=False)
    bad_words.to_csv(out_dir / "bad_words.csv", index=False)
    if not log_df.empty:
        log_df.to_csv(out_dir / "training_log_summary.csv", index=False)

    cm_path = save_confusion_matrix(df, out_dir)
    curve_paths = save_training_curves(log_df, out_dir)

    # -----------------------------
    # Write summary txt
    # -----------------------------
    summary_lines = []
    summary_lines.append("RUN ANALYSIS SUMMARY")
    summary_lines.append("=" * 60)
    summary_lines.append(f"Run directory: {run_dir}")
    summary_lines.append(f"Prediction CSV: {pred_csv}")
    summary_lines.append(f"Output directory: {out_dir}")
    summary_lines.append("")
    summary_lines.append(f"Total evaluated samples: {len(df)}")
    summary_lines.append(f"Overall accuracy: {overall_acc:.4f}")
    summary_lines.append(f"Number of classes/words: {df['true_label_name'].nunique()}")
    summary_lines.append("")

    if best_epoch_info:
        summary_lines.append(f"Best epoch by accuracy: {best_epoch_info['best_epoch_by_accuracy']}")
        summary_lines.append(f"Best accuracy from log: {best_epoch_info['best_accuracy']:.4f}")
        summary_lines.append("")

    if not good_words.empty:
        summary_lines.append(f"Top {min(args.top_k, len(good_words))} GOOD words:")
        for _, row in good_words.iterrows():
            summary_lines.append(
                f"  - {row['word']}: acc={row['accuracy']:.4f}, "
                f"correct={row['correct']}/{row['total_samples']}"
            )
        summary_lines.append("")

    if not bad_words.empty:
        summary_lines.append(f"Top {min(args.top_k, len(bad_words))} BAD words:")
        for _, row in bad_words.iterrows():
            summary_lines.append(
                f"  - {row['word']}: acc={row['accuracy']:.4f}, "
                f"correct={row['correct']}/{row['total_samples']}, "
                f"confused_with={row['top_confusions']}"
            )
        summary_lines.append("")

    summary_lines.append("Saved files:")
    summary_lines.append(f"  - {out_dir / 'classification_report.csv'}")
    summary_lines.append(f"  - {out_dir / 'per_word_stats.csv'}")
    summary_lines.append(f"  - {out_dir / 'good_words.csv'}")
    summary_lines.append(f"  - {out_dir / 'bad_words.csv'}")
    summary_lines.append(f"  - {cm_path}")
    for p in curve_paths:
        summary_lines.append(f"  - {p}")

    summary_txt = "\n".join(summary_lines)
    with open(out_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_txt)

    print(summary_txt)


if __name__ == "__main__":
    main()