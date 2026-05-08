import ast
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix

mpl.rcParams["font.family"] = "DejaVu Sans"
mpl.rcParams["axes.unicode_minus"] = False

try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    def ar_text(s):
        s = str(s)
        return get_display(arabic_reshaper.reshape(s))
except Exception:
    def ar_text(s):
        return str(s)

RUN_DIR = Path("/home/sign_lang_fyp_sp26/fyp/experiments/osl_words_1gpu_20260417_013329")
ANALYSIS_DIR = RUN_DIR / "analysis"
OUT_DIR = RUN_DIR / "analysis_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PREDICTIONS_CSV = ANALYSIS_DIR / "predictions_with_names.csv"
PER_WORD_CSV = ANALYSIS_DIR / "per_word_stats.csv"
GOOD_WORDS_CSV = ANALYSIS_DIR / "good_words.csv"
BAD_WORDS_CSV = ANALYSIS_DIR / "bad_words.csv"
CLASS_REPORT_CSV = ANALYSIS_DIR / "classification_report.csv"
TRAIN_LOG_SUMMARY_CSV = ANALYSIS_DIR / "training_log_summary.csv"
SUMMARY_TXT = ANALYSIS_DIR / "summary.txt"

df = pd.read_csv(PREDICTIONS_CSV, encoding="utf-8")
per_word = pd.read_csv(PER_WORD_CSV, encoding="utf-8")
good_words = pd.read_csv(GOOD_WORDS_CSV, encoding="utf-8")
bad_words = pd.read_csv(BAD_WORDS_CSV, encoding="utf-8")
class_report = pd.read_csv(CLASS_REPORT_CSV, encoding="utf-8")
train_log_summary = pd.read_csv(TRAIN_LOG_SUMMARY_CSV, encoding="utf-8") if TRAIN_LOG_SUMMARY_CSV.exists() else pd.DataFrame()

summary_text = SUMMARY_TXT.read_text(encoding="utf-8") if SUMMARY_TXT.exists() else ""

def parse_confusions(x):
    if pd.isna(x):
        return {}
    x = str(x).strip()
    if not x:
        return {}
    try:
        return ast.literal_eval(x)
    except Exception:
        return {"raw": x}

if "top_confusions" in per_word.columns:
    per_word["top_confusions"] = per_word["top_confusions"].apply(parse_confusions)

for name, frame in [
    ("predictions_with_names_utf8.csv", df),
    ("per_word_stats_utf8.csv", per_word),
    ("good_words_utf8.csv", good_words),
    ("bad_words_utf8.csv", bad_words),
    ("classification_report_utf8.csv", class_report),
]:
    frame.to_csv(OUT_DIR / name, index=False, encoding="utf-8-sig")

xlsx_path = OUT_DIR / "analysis_results.xlsx"
with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
    df.to_excel(writer, sheet_name="predictions", index=False)
    per_word.to_excel(writer, sheet_name="per_word_stats", index=False)
    good_words.to_excel(writer, sheet_name="good_words", index=False)
    bad_words.to_excel(writer, sheet_name="bad_words", index=False)
    class_report.to_excel(writer, sheet_name="classification_report", index=False)
    if not train_log_summary.empty:
        train_log_summary.to_excel(writer, sheet_name="training_log", index=False)

best10 = per_word.sort_values(["accuracy", "total_samples"], ascending=[False, False]).head(10).copy()
best10 = best10.iloc[::-1]

plt.figure(figsize=(10, 6))
plt.barh([ar_text(x) for x in best10["word"]], best10["accuracy"])
plt.xlabel("Accuracy")
plt.ylabel("Word")
plt.title("Top 10 Best Recognized Words")
plt.tight_layout()
plt.savefig(OUT_DIR / "best_words_bar.png", dpi=220)
plt.close()

worst10 = per_word.sort_values(["accuracy", "total_samples"], ascending=[True, False]).head(10).copy()
worst10 = worst10.iloc[::-1]

plt.figure(figsize=(10, 6))
plt.barh([ar_text(x) for x in worst10["word"]], worst10["accuracy"])
plt.xlabel("Accuracy")
plt.ylabel("Word")
plt.title("Top 10 Worst Recognized Words")
plt.tight_layout()
plt.savefig(OUT_DIR / "worst_words_bar.png", dpi=220)
plt.close()

wrong = df[df["true_label_name"] != df["pred_label_name"]].copy()
pairs = (
    wrong.groupby(["true_label_name", "pred_label_name"])
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
    .head(10)
    .copy()
)

pairs["pair"] = pairs.apply(
    lambda r: f"{ar_text(r['true_label_name'])} → {ar_text(r['pred_label_name'])}",
    axis=1
)
pairs = pairs.iloc[::-1]

plt.figure(figsize=(12, 7))
plt.barh(pairs["pair"], pairs["count"])
plt.xlabel("Count")
plt.ylabel("Confusion Pair")
plt.title("Top 10 Most Frequent Confusions")
plt.tight_layout()
plt.savefig(OUT_DIR / "top_confusions_bar.png", dpi=220)
plt.close()

pairs.to_csv(OUT_DIR / "top_confusion_pairs_utf8.csv", index=False, encoding="utf-8-sig")

labels = sorted(df["true_label_name"].astype(str).unique().tolist())
cm = confusion_matrix(df["true_label_name"], df["pred_label_name"], labels=labels)

fig_size = max(14, min(24, int(len(labels) * 0.6)))
plt.figure(figsize=(fig_size, fig_size))
plt.imshow(cm, interpolation="nearest", aspect="auto")
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = range(len(labels))
plt.xticks(tick_marks, [ar_text(x) for x in labels], rotation=90, fontsize=10)
plt.yticks(tick_marks, [ar_text(x) for x in labels], fontsize=10)
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.tight_layout()
plt.savefig(OUT_DIR / "confusion_matrix_arabic.png", dpi=240, bbox_inches="tight")
plt.close()

summary_lines = []
summary_lines.append("ANALYSIS V2 SUMMARY")
summary_lines.append("=" * 60)
summary_lines.append(f"Run directory: {RUN_DIR}")
summary_lines.append(f"Source analysis: {ANALYSIS_DIR}")
summary_lines.append(f"Output directory: {OUT_DIR}")
summary_lines.append("")
summary_lines.append("Created files:")
summary_lines.append(f"- {OUT_DIR / 'analysis_results.xlsx'}")
summary_lines.append(f"- {OUT_DIR / 'best_words_bar.png'}")
summary_lines.append(f"- {OUT_DIR / 'worst_words_bar.png'}")
summary_lines.append(f"- {OUT_DIR / 'top_confusions_bar.png'}")
summary_lines.append(f"- {OUT_DIR / 'confusion_matrix_arabic.png'}")
summary_lines.append("")
summary_lines.append("Top 10 best words:")
for _, row in best10.iloc[::-1].iterrows():
    summary_lines.append(
        f"  - {row['word']}: acc={row['accuracy']:.4f}, correct={int(row['correct'])}/{int(row['total_samples'])}"
    )

summary_lines.append("")
summary_lines.append("Top 10 worst words:")
for _, row in worst10.iterrows():
    summary_lines.append(
        f"  - {row['word']}: acc={row['accuracy']:.4f}, correct={int(row['correct'])}/{int(row['total_samples'])}"
    )

summary_lines.append("")
summary_lines.append("Top 10 confusion pairs:")
for _, row in pairs.iloc[::-1].iterrows():
    summary_lines.append(
        f"  - {row['true_label_name']} -> {row['pred_label_name']}: {int(row['count'])}"
    )

(OUT_DIR / "summary_v2.txt").write_text("\n".join(summary_lines), encoding="utf-8")

print("Done.")
print(f"Saved Excel workbook: {xlsx_path}")
print(f"Saved outputs in: {OUT_DIR}")
