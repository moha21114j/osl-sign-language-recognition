"""
Live training metrics monitor.

Usage:
    python monitor_training.py              # monitor words (default)
    python monitor_training.py sentences    # monitor sentences
    python monitor_training.py all          # monitor both

Re-run at any time to see updated results.
"""
import json, sys, datetime
from pathlib import Path
from training_config import UNISIGN_DIR

LOG_PATHS = {
    "words":     UNISIGN_DIR / "out" / "osl_words_finetuning" / "log.txt",
    "sentences": UNISIGN_DIR / "out" / "osl_sentences_finetuning" / "log.txt",
}

RUN_LOG = Path(__file__).resolve().parent / "training_runs.log"

def display_log(name: str, log_path: Path):
    header = f"OSL-{'Words (ISLR)' if name == 'words' else 'Sentences (SLT)'}"
    print("=" * 80)
    print(f"  {header}")
    print(f"  Log: {log_path}")
    print("=" * 80)

    if not log_path.exists():
        print("  (no log file yet — training has not started)\n")
        return

    entries = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    if not entries:
        print("  (log file is empty)\n")
        return

    # Split into runs when epoch resets
    runs = []
    current_run = []
    for e in entries:
        if current_run and e["epoch"] <= current_run[-1]["epoch"]:
            runs.append(current_run)
            current_run = []
        current_run.append(e)
    if current_run:
        runs.append(current_run)

    for run_idx, run_entries in enumerate(runs, 1):
        params = run_entries[0].get("n_parameters", "?")
        n_epochs = len(run_entries)
        is_latest = (run_idx == len(runs))
        tag = " (LATEST / ACTIVE)" if is_latest else " (old run)"

        print(f"\n  Run {run_idx}{tag} -- {params}M params -- {n_epochs} epochs")
        print(f"  {'Epoch':>5}  {'Train Loss':>10}  {'Dev Loss':>10}  {'PI Acc %':>9}  {'PC Acc %':>9}  {'LR':>12}")
        print(f"  {'-----':>5}  {'----------':>10}  {'--------':>10}  {'---------':>9}  {'---------':>9}  {'----------':>12}")

        best_acc = 0
        best_epoch = -1
        for e in run_entries:
            pi = e.get("test_top1_acc_pi", e.get("test_bleu4", "N/A"))
            pc = e.get("test_top1_acc_pc", e.get("test_bleu1", "N/A"))
            ep = e["epoch"]

            marker = ""
            if isinstance(pi, (int, float)) and pi > best_acc:
                best_acc = pi
                best_epoch = ep
                marker = " <-- best"

            pi_s = f"{pi:.2f}" if isinstance(pi, (int, float)) else str(pi)
            pc_s = f"{pc:.2f}" if isinstance(pc, (int, float)) else str(pc)
            print(f"  {ep:>5}  {e['train_loss']:>10.3f}  {e['test_loss']:>10.3f}  {pi_s:>9}  {pc_s:>9}  {e['train_lr']:>12.6f}{marker}")

        if best_epoch >= 0:
            print(f"\n  >>> Best accuracy: {best_acc:.2f}% at epoch {best_epoch}")

        if is_latest:
            target = 20
            remaining = target - n_epochs
            if remaining > 0:
                print(f"  >>> Progress: {n_epochs}/{target} epochs, ~{remaining} remaining")
            else:
                print(f"  >>> Training complete: {n_epochs}/{target} epochs")
                # Append to run log
                entry = (f"{datetime.datetime.now():%Y-%m-%d %H:%M} | {header:25s} | "
                         f"{n_epochs} epochs | Best: {best_acc:.2f}% | {params}M params\n")
                with open(RUN_LOG, "a") as rl:
                    rl.write(entry)
                print(f"  (logged to {RUN_LOG.name})")
    print()


if __name__ == "__main__":
    target = sys.argv[1].lower() if len(sys.argv) > 1 else "all"

    if target == "all":
        for name, path in LOG_PATHS.items():
            display_log(name, path)
    elif target in LOG_PATHS:
        display_log(target, LOG_PATHS[target])
    else:
        print(f"Unknown target: {target}")
        print("Usage: python monitor_training.py [words|sentences|all]")
        sys.exit(1)
