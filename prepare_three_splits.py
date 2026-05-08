"""
OSL-Sentences — Three Split Preparation Script
===============================================
Generates 3 splits from the existing combined labels:

  Split A — Sentence independent
    - Sentences randomly divided 70/15/15 across train/dev/test
    - All signers appear in all splits
    - Zero sentence text overlap between any split

  Split B — Signer independent
    - Test:  S18 only
    - Dev:   S03 only
    - Train: all remaining signers
    - Sentences can overlap across splits

  Split C — No overlap (strictest)
    - Test:  S18 only + sentences that ONLY S18 signed
    - Dev:   S03 only + sentences that ONLY S03 signed
    - Train: remaining signers + remaining sentences only
    - Zero sentence AND zero signer overlap

Usage:
  cd ~/fyp/Uni-sign-main/Uni-Sign-main/
  python3 prepare_three_splits.py

Output label files:
  data/OSL-Sentences-splitA/labels-osl.{train,dev,test}
  data/OSL-Sentences-splitB/labels-osl.{train,dev,test}
  data/OSL-Sentences-splitC/labels-osl.{train,dev,test}
"""

import gzip
import pickle
import re
import random
import os
from collections import defaultdict
from pathlib import Path

SEED = 42
random.seed(SEED)

BASE_LABELS = 'data/OSL-Sentences'
OUT_A = 'data/OSL-Sentences-splitA'
OUT_B = 'data/OSL-Sentences-splitB'
OUT_C = 'data/OSL-Sentences-splitC'

TEST_SIGNER  = 'S18'
DEV_SIGNER   = 'S03'

# ── helpers ──────────────────────────────────────────────────────────────────

def load_all():
    """Load and merge all existing splits into one dict."""
    all_data = {}
    for split in ['train', 'dev', 'test']:
        with gzip.open(f'{BASE_LABELS}/labels-osl.{split}', 'rb') as f:
            d = pickle.load(f)
        all_data.update(d)
    print(f'Loaded {len(all_data)} total samples')
    return all_data

def get_signer(entry):
    m = re.search(r'S\d+', entry.get('video_path', ''))
    return m.group() if m else None

def save_split(data_dict, out_dir, split_name):
    os.makedirs(out_dir, exist_ok=True)
    out_path = f'{out_dir}/labels-osl.{split_name}'
    with gzip.open(out_path, 'wb') as f:
        pickle.dump(data_dict, f)
    print(f'  Saved {len(data_dict):5d} samples → {out_path}')

def print_stats(name, train, dev, test):
    print(f'\n{"="*60}')
    print(f'Split {name}')
    print(f'{"="*60}')
    for split_name, d in [('train', train), ('dev', dev), ('test', test)]:
        signers = sorted(set(get_signer(v) for v in d.values() if get_signer(v)))
        sents   = set(v.get('text','') for v in d.values())
        print(f'  {split_name:5s}: {len(d):5d} samples | {len(sents):3d} unique sentences | signers: {signers}')

    # overlap checks
    train_sents = set(v.get('text','') for v in train.values())
    dev_sents   = set(v.get('text','') for v in dev.values())
    test_sents  = set(v.get('text','') for v in test.values())
    train_signers = set(get_signer(v) for v in train.values())
    dev_signers   = set(get_signer(v) for v in dev.values())
    test_signers  = set(get_signer(v) for v in test.values())

    print(f'\n  Sentence overlap:')
    print(f'    test  ∩ train : {len(test_sents  & train_sents)}')
    print(f'    test  ∩ dev   : {len(test_sents  & dev_sents)}')
    print(f'    dev   ∩ train : {len(dev_sents   & train_sents)}')
    print(f'\n  Signer overlap:')
    print(f'    test  ∩ train : {sorted(test_signers  & train_signers)}')
    print(f'    test  ∩ dev   : {sorted(test_signers  & dev_signers)}')
    print(f'    dev   ∩ train : {sorted(dev_signers   & train_signers)}')

# ── Split A — Sentence independent ───────────────────────────────────────────

def make_split_A(all_data):
    print('\nBuilding Split A — Sentence independent...')

    # group all entries by sentence text
    by_sentence = defaultdict(dict)
    for k, v in all_data.items():
        by_sentence[v.get('text', '')][k] = v

    sentences = list(by_sentence.keys())
    random.shuffle(sentences)

    n = len(sentences)
    n_test = max(1, int(round(n * 0.15)))
    n_dev  = max(1, int(round(n * 0.15)))
    n_train = n - n_test - n_dev

    train_sents = set(sentences[:n_train])
    dev_sents   = set(sentences[n_train:n_train + n_dev])
    test_sents  = set(sentences[n_train + n_dev:])

    train, dev, test = {}, {}, {}
    for k, v in all_data.items():
        s = v.get('text', '')
        if s in train_sents:
            train[k] = v
        elif s in dev_sents:
            dev[k] = v
        elif s in test_sents:
            test[k] = v

    print_stats('A (Sentence independent)', train, dev, test)
    save_split(train, OUT_A, 'train')
    save_split(dev,   OUT_A, 'dev')
    save_split(test,  OUT_A, 'test')

# ── Split B — Signer independent ─────────────────────────────────────────────

def make_split_B(all_data):
    print('\nBuilding Split B — Signer independent...')

    train, dev, test = {}, {}, {}
    for k, v in all_data.items():
        s = get_signer(v)
        if s == TEST_SIGNER:
            test[k] = v
        elif s == DEV_SIGNER:
            dev[k] = v
        else:
            train[k] = v

    print_stats('B (Signer independent)', train, dev, test)
    save_split(train, OUT_B, 'train')
    save_split(dev,   OUT_B, 'dev')
    save_split(test,  OUT_B, 'test')

# ── Split C — No overlap ──────────────────────────────────────────────────────

def make_split_C(all_data):
    print('\nBuilding Split C — No overlap (strictest)...')

    # Find which sentences each signer signed
    signer_sentences = defaultdict(set)
    for v in all_data.values():
        s = get_signer(v)
        if s:
            signer_sentences[s].add(v.get('text', ''))

    all_signers = set(signer_sentences.keys())
    other_signers = all_signers - {TEST_SIGNER, DEV_SIGNER}

    # Sentences signed by other signers (i.e. would appear in train)
    train_signer_sents = set()
    for s in other_signers:
        train_signer_sents |= signer_sentences[s]

    # Exclusive sentences: only signed by test/dev signer, not by anyone in train
    test_exclusive_sents = signer_sentences[TEST_SIGNER] - train_signer_sents - signer_sentences[DEV_SIGNER]
    dev_exclusive_sents  = signer_sentences[DEV_SIGNER]  - train_signer_sents - signer_sentences[TEST_SIGNER]

    print(f'\n  Exclusive sentences for {TEST_SIGNER} (test): {len(test_exclusive_sents)}')
    print(f'  Exclusive sentences for {DEV_SIGNER}  (dev):  {len(dev_exclusive_sents)}')

    # Build splits
    train, dev, test = {}, {}, {}
    skipped = 0
    for k, v in all_data.items():
        s    = get_signer(v)
        sent = v.get('text', '')

        if s == TEST_SIGNER:
            if sent in test_exclusive_sents:
                test[k] = v
            else:
                skipped += 1  # S18 video but sentence not exclusive
        elif s == DEV_SIGNER:
            if sent in dev_exclusive_sents:
                dev[k] = v
            else:
                skipped += 1  # S03 video but sentence not exclusive
        else:
            # Train: only sentences NOT in test or dev exclusive sets
            if sent not in test_exclusive_sents and sent not in dev_exclusive_sents:
                train[k] = v
            # else: this sentence is reserved for test/dev — exclude from train

    print(f'  Skipped {skipped} samples (signer matches but sentence not exclusive)')
    print_stats('C (No overlap)', train, dev, test)
    save_split(train, OUT_C, 'train')
    save_split(dev,   OUT_C, 'dev')
    save_split(test,  OUT_C, 'test')

# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    all_data = load_all()

    # print overall stats
    signer_counts = defaultdict(int)
    for v in all_data.values():
        s = get_signer(v)
        if s:
            signer_counts[s] += 1
    print('\nAll signers:')
    for s in sorted(signer_counts):
        print(f'  {s}: {signer_counts[s]} videos')

    make_split_A(all_data)
    make_split_B(all_data)
    make_split_C(all_data)

    print('\n\nDone! Label files written to:')
    print(f'  {OUT_A}/labels-osl.{{train,dev,test}}')
    print(f'  {OUT_B}/labels-osl.{{train,dev,test}}')
    print(f'  {OUT_C}/labels-osl.{{train,dev,test}}')
    print('\nNext steps:')
    print('  1. Update config.py to add paths for splitA, splitB, splitC')
    print('  2. Launch 3 training runs on separate GPUs')
    print('  3. Pose files stay in dataset/OSL-Sentences/ — no copying needed')
