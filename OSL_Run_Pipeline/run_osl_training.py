"""
Standalone training script for OSL-Words fine-tuning on Uni-Sign.
Run from the Uni-Sign-main/Uni-Sign-main/ directory:

    python ../../OSL_FineTuning/run_osl_training.py

Or copy into Uni-Sign-main/Uni-Sign-main/ and run directly:

    python run_osl_training.py
"""
import os, sys, argparse
from pathlib import Path

# Ensure we're in the Uni-Sign directory
SCRIPT_DIR = Path(__file__).resolve().parent
UNISIGN_DIR = SCRIPT_DIR.parent.parent / "Uni-Sign-main"

os.chdir(str(UNISIGN_DIR))
if str(UNISIGN_DIR) not in sys.path:
    sys.path.insert(0, str(UNISIGN_DIR))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.argv = [
    "fine_tuning.py",
    "--batch-size",                  "4",
    "--epochs",                      "20",
    "--opt",                         "AdamW",
    "--lr",                          "3e-4",
    "--warmup-epochs",               "2",
    "--output_dir",                  "checkpoints/osl_words_finetuning",
    "--finetune",                    "checkpoints/csl_stage2_weight.pth",
    "--dataset",                     "OSL-Words",
    "--task",                        "ISLR",
    "--max_length",                  "256",
    "--zero_stage",                  "0",
    "--num_workers",                 "4",
]

if __name__ == '__main__':
    from utils import get_args_parser
    import fine_tuning

    parser = argparse.ArgumentParser("Uni-Sign OSL", parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    fine_tuning.main(args)