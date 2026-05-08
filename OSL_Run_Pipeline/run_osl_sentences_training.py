"""
Standalone training script for OSL-Sentences fine-tuning on Uni-Sign.
Run from the Uni-Sign-main/Uni-Sign-main/ directory:

    python ../../OSL_FineTuning/run_osl_sentences_training.py

Or copy into Uni-Sign-main/Uni-Sign-main/ and run directly:

    python run_osl_sentences_training.py
"""
import os, sys, argparse, re
from pathlib import Path

# Ensure we're in the Uni-Sign directory
SCRIPT_DIR = Path(__file__).resolve().parent
UNISIGN_DIR = SCRIPT_DIR.parent.parent / "Uni-Sign-main" / "Uni-Sign-main"

os.chdir(str(UNISIGN_DIR))
if str(UNISIGN_DIR) not in sys.path:
    sys.path.insert(0, str(UNISIGN_DIR))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

OUTPUT_DIR = UNISIGN_DIR / "checkpoints" / "osl_sentences_finetuning"
PRETRAINED_CKP = UNISIGN_DIR / "checkpoints" / "csl_stage2_weight.pth"
TOTAL_EPOCHS = 20


def find_latest_sentences_checkpoint(output_dir: Path):
    pattern = re.compile(r"checkpoint_(\d+)\.pth$")
    latest_epoch = -1
    latest_ckpt = None
    for ckpt in output_dir.glob("checkpoint_*.pth"):
        m = pattern.match(ckpt.name)
        if m:
            epoch = int(m.group(1))
            if epoch > latest_epoch:
                latest_epoch = epoch
                latest_ckpt = ckpt
    return latest_ckpt, latest_epoch


def build_argv():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    resume_ckpt, last_epoch = find_latest_sentences_checkpoint(OUTPUT_DIR)

    if resume_ckpt is not None and last_epoch + 1 < TOTAL_EPOCHS:
        finetune_ckpt = str(resume_ckpt)
        start_epoch = last_epoch + 1
        warmup_epochs = 0
        print(f"Resuming Sentences from {resume_ckpt.name} (start_epoch={start_epoch}, total_epochs={TOTAL_EPOCHS})")
    else:
        finetune_ckpt = str(PRETRAINED_CKP)
        start_epoch = 0
        warmup_epochs = 2
        if resume_ckpt is not None and last_epoch + 1 >= TOTAL_EPOCHS:
            print(f"Detected completed Sentences run up to epoch {last_epoch}. Starting fresh run from pretrained checkpoint.")
        else:
            print("No prior Sentences checkpoints found. Starting from pretrained checkpoint.")

    return [
        "fine_tuning.py",
        "--batch-size",                  "2",
        "--gradient-accumulation-steps", "4",
        "--epochs",                      str(TOTAL_EPOCHS),
        "--start-epoch",                 str(start_epoch),
        "--opt",                         "AdamW",
        "--lr",                          "1e-4",
        "--warmup-epochs",               str(warmup_epochs),
        "--output_dir",                  str(OUTPUT_DIR),
        "--finetune",                    finetune_ckpt,
        "--dataset",                     "OSL-Sentences",
        "--task",                        "SLT",
        "--max_length",                  "128",
        "--zero_stage",                  "0",
        "--num_workers",                 "4",
    ]

if __name__ == '__main__':
    from utils import get_args_parser
    import fine_tuning

    sys.argv = build_argv()

    parser = argparse.ArgumentParser("Uni-Sign OSL Sentences", parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    fine_tuning.main(args)