#!/usr/bin/env python3
import argparse
import csv
import gzip
import os
import pickle
import random
import shutil
from collections import defaultdict
from pathlib import Path


def dump_gzip_pickle(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def clear_directory(path: Path):
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        return
    for child in path.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def safe_link(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def read_metadata(metadata_path: Path):
    rows = []
    with metadata_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            normalized_row = {key.strip(): value for key, value in row.items() if key is not None}
            rows.append(normalized_row)
    if not rows:
        raise RuntimeError(f"No rows found in metadata: {metadata_path}")
    return rows


def build_groups(rows):
    grouped = defaultdict(list)
    for row in rows:
        base_video_id = row["video_id"]
        item_id = row["item_id"]
        speaker = row["speaker"]
        grouped[(item_id, speaker)].append(
            {
                "base_video_id": base_video_id,
                "item_id": item_id,
                "speaker": speaker,
                "take": row["take"],
                "label": row["label"].strip(),
                "rgb_file": row["filename"],
                "pose_file": f"{base_video_id}.pkl",
            }
        )
    return grouped


def split_groups(group_keys, seed, train_ratio, dev_ratio):
    random.Random(seed).shuffle(group_keys)
    total = len(group_keys)
    if total < 3:
        raise RuntimeError("Need at least 3 sentence speaker groups to create train/dev/test splits")

    n_train = max(1, int(round(total * train_ratio)))
    n_dev = max(1, int(round(total * dev_ratio)))
    if n_train + n_dev >= total:
        n_dev = max(1, total - n_train - 1)
    n_test = total - n_train - n_dev
    if n_test <= 0:
        n_test = 1
        if n_train > n_dev:
            n_train -= 1
        else:
            n_dev -= 1

    return {
        "train": group_keys[:n_train],
        "dev": group_keys[n_train:n_train + n_dev],
        "test": group_keys[n_train + n_dev:],
    }


def collect_augmented_variants(base_video_id, augmented_pose_dir: Path, augmented_rgb_dir: Path):
    variants = []
    for pose_file in sorted(augmented_pose_dir.glob(f"*_{base_video_id}.pkl")):
        rgb_file = augmented_rgb_dir / f"{pose_file.stem}.mp4"
        if rgb_file.exists():
            variants.append({
                "variant_name": pose_file.stem,
                "pose_path": pose_file,
                "rgb_path": rgb_file,
            })
    return variants


def build_entries(split_name, group_keys, grouped_rows, args, out_dirs):
    entries = {}
    linked_count = 0

    for group_key in group_keys:
        samples = sorted(grouped_rows[group_key], key=lambda sample: sample["take"])
        for sample in samples:
            base_rgb = args.original_rgb_dir / sample["rgb_file"]
            base_pose = args.original_pose_dir / sample["pose_file"]
            if not base_rgb.exists() or not base_pose.exists():
                raise FileNotFoundError(f"Missing base files for {sample['base_video_id']}")

            base_name = sample["base_video_id"]
            entries[f"{split_name}_{base_name}"] = {
                "name": base_name,
                "text": sample["label"],
                "gloss": [],
                "video_path": f"{base_name}.mp4",
            }
            safe_link(base_rgb, out_dirs["rgb"] / split_name / f"{base_name}.mp4")
            safe_link(base_pose, out_dirs["pose"] / split_name / f"{base_name}.pkl")
            linked_count += 1

            if split_name != "train" or not args.include_augmented:
                continue

            for variant in collect_augmented_variants(base_name, args.augmented_pose_dir, args.augmented_rgb_dir):
                variant_name = variant["variant_name"]
                entries[f"train_{variant_name}"] = {
                    "name": variant_name,
                    "text": sample["label"],
                    "gloss": [],
                    "video_path": f"{variant_name}.mp4",
                }
                safe_link(variant["rgb_path"], out_dirs["rgb"] / "train" / f"{variant_name}.mp4")
                safe_link(variant["pose_path"], out_dirs["pose"] / "train" / f"{variant_name}.pkl")
                linked_count += 1

    return entries, linked_count


def main():
    parser = argparse.ArgumentParser(description="Prepare OSL-Sentences for Uni-Sign fine-tuning")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--dev-ratio", type=float, default=0.15)
    parser.add_argument("--no-augmented", action="store_true", help="Exclude augmented sentence variants from the training split")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    dataset_root = script_dir.parent
    uni_root = dataset_root.parent / "Uni-Sign-main" / "Uni-Sign-main"

    metadata_path = dataset_root / "data" / "OSL-Sentences" / "metadata.csv"
    args.original_rgb_dir = dataset_root / "dataset" / "OSL-Sentences" / "rgb_format"
    args.original_pose_dir = dataset_root / "dataset" / "OSL-Sentences" / "pose_format"
    args.augmented_rgb_dir = dataset_root / "videos_aug" / "OSL-Sentences" / "rgb_format"
    args.augmented_pose_dir = dataset_root / "videos_aug" / "OSL-Sentences" / "pose_format"
    args.include_augmented = not args.no_augmented

    out_dirs = {
        "rgb": uni_root / "dataset" / "OSL-Sentences" / "rgb_format",
        "pose": uni_root / "dataset" / "OSL-Sentences" / "pose_format",
        "labels": uni_root / "data" / "OSL-Sentences",
    }

    rows = read_metadata(metadata_path)
    grouped_rows = build_groups(rows)
    split_keys = split_groups(list(grouped_rows.keys()), args.seed, args.train_ratio, args.dev_ratio)

    for split_name in ("train", "dev", "test"):
        clear_directory(out_dirs["rgb"] / split_name)
        clear_directory(out_dirs["pose"] / split_name)

    stats = {}
    for split_name, group_keys in split_keys.items():
        entries, linked_count = build_entries(split_name, group_keys, grouped_rows, args, out_dirs)
        dump_gzip_pickle(entries, out_dirs["labels"] / f"labels-osl.{split_name}")
        stats[split_name] = {
            "groups": len(group_keys),
            "entries": len(entries),
            "linked_files": linked_count,
        }

    print("Prepared OSL-Sentences for Uni-Sign")
    for split_name in ("train", "dev", "test"):
        split_stats = stats[split_name]
        print(
            f"  {split_name}: groups={split_stats['groups']}, "
            f"entries={split_stats['entries']}, linked={split_stats['linked_files']}"
        )
    print(f"  augmented_in_train={args.include_augmented}")
    print(f"  label_dir={out_dirs['labels']}")


if __name__ == "__main__":
    main()