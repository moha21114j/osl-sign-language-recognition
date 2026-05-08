# ============================================================
#  OSL Training Configuration
#  Edit these values and they will be used by the training
#  scripts. Import this file instead of hard-coding args.
# ============================================================

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────
UNISIGN_DIR    = Path(r"C:\Users\s132606\Downloads\FYPproject\Uni-sign-main\Uni-Sign-main\Uni-Sign-main")
PRETRAINED_CKP = UNISIGN_DIR / "checkpoints" / "csl_stage2_weight.pth"

# ── OSL-Words (ISLR) ──────────────────────────────────────
WORDS_CONFIG = {
    "dataset":                     "OSL-Words",
    "task":                        "ISLR",
    "batch_size":                  4,
    "gradient_accumulation_steps": 2,
    "epochs":                      20,
    "optimizer":                   "AdamW",
    "lr":                          3e-4,
    "warmup_epochs":               2,
    "label_smoothing":             0.2,       # set in Uni-Sign defaults
    "max_length":                  64,
    "zero_stage":                  0,
    "num_workers":                 4,
    "output_dir":                  str(UNISIGN_DIR / "checkpoints" / "osl_words_finetuning"),
    "finetune":                    str(PRETRAINED_CKP),
}

# ── OSL-Sentences (SLT) ──────────────────────────────────
SENTENCES_CONFIG = {
    "dataset":                     "OSL-Sentences",
    "task":                        "SLT",
    "batch_size":                  2,
    "gradient_accumulation_steps": 4,
    "epochs":                      20,
    "optimizer":                   "AdamW",
    "lr":                          1e-4,
    "warmup_epochs":               2,
    "label_smoothing":             0.2,
    "max_length":                  128,
    "zero_stage":                  0,
    "num_workers":                 4,
    "output_dir":                  str(UNISIGN_DIR / "checkpoints" / "osl_sentences_finetuning"),
    "finetune":                    str(PRETRAINED_CKP),
}

# ── Hardware Notes ─────────────────────────────────────────
# GPU: NVIDIA GeForce RTX 4060 Ti (~8 GB VRAM)
# Torch: 2.6.0+cu124, bf16 training
# Estimated time: ~18h words + ~18h sentences = ~36h total
