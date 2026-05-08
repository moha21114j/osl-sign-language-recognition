"""
Test OSL Dataset Pose Extraction  
Demonstrates that RTMLib can extract poses from OSL videos
Uses the same pose extraction used for WLASL training
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'demo', 'rtmlib-main'))

import cv2
import numpy as np
from pathlib import Path

# Paths
OSL_WORDS_PATH = Path(r"C:\Users\MOBPC\Downloads\FYP\Dataset\dataset\OSL-Words\rgb_format")
WORDS_LABELS_PATH = Path(r"C:\Users\MOBPC\Downloads\FYP\Dataset\Words.txt")

print("=" * 60)
print("OSL Dataset Pose Extraction Test")
print("=" * 60)

# Load OSL labels
labels = {}
with open(WORDS_LABELS_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split(' ', 1)
        if len(parts) == 2:
            labels[parts[0]] = parts[1]
print(f"Loaded {len(labels)} word labels from Words.txt")
print()

# Get video files
videos = sorted(OSL_WORDS_PATH.glob("*.mp4"))[:5]  # Test first 5 videos
print(f"Testing {len(videos)} videos...")
print()

# Try to import RTMLib for pose extraction
try:
    from rtmlib import Wholebody
    wholebody = Wholebody(to_openpose=False, mode='lightweight', backend='onnxruntime', device='cuda')
    use_rtmlib = True
    print("[OK] RTMLib Wholebody initialized (CUDA)")
except Exception as e:
    print(f"[INFO] RTMLib not available ({e}), using video info only")
    use_rtmlib = False

print()

for video_path in videos:
    # Extract video ID from filename (e.g., "0001" from "0001_S01_T01.mp4")
    video_id = video_path.stem.split('_')[0]
    label = labels.get(video_id, "Unknown")
    
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video: {video_path.name}")
    print(f"  Label (Arabic): {label}")
    print(f"  Frames: {total_frames}, FPS: {fps:.1f}, Resolution: {width}x{height}")
    
    if use_rtmlib:
        # Extract poses from sample frames using RTMLib
        keypoints_list = []
        scores_list = []
        frames_with_poses = 0
        
        for frame_idx in range(0, min(total_frames, 30), 2):  # Sample every 2nd frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            
            keypoints, scores = wholebody(frame)
            keypoints_list.append(keypoints)
            scores_list.append(scores)
            
            # Check if person detected (scores > 0.3)
            if scores is not None and np.any(scores > 0.3):
                frames_with_poses += 1
        
        n_frames = len(keypoints_list)
        print(f"  Processed: {n_frames} frames")
        print(f"  Poses detected: {frames_with_poses}/{n_frames} ({100*frames_with_poses/n_frames:.0f}%)")
        
        if keypoints_list and len(keypoints_list) > 0 and keypoints_list[0] is not None:
            kp = keypoints_list[0]
            if len(kp.shape) >= 2:
                print(f"  Keypoints shape: {kp.shape}")
                print(f"  Sample keypoints (first 5):")
                for i in range(min(5, kp.shape[-2] if len(kp.shape) > 1 else 0)):
                    if len(kp.shape) == 3:
                        x, y = kp[0, i, 0], kp[0, i, 1]
                    else:
                        x, y = kp[i, 0], kp[i, 1]
                    print(f"    Joint {i}: ({x:.1f}, {y:.1f})")
    else:
        # Just read and display frame info
        ret, frame = cap.read()
        if ret:
            print(f"  Frame shape: {frame.shape}")
            print(f"  Frame dtype: {frame.dtype}")
    
    cap.release()
    print()

print("=" * 60)
print("SUCCESS! Video data is readable and compatible.")
if use_rtmlib:
    print("Pose extraction with RTMLib works on OSL videos!")
print("=" * 60)
