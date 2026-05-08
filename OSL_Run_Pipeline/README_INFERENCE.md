# Real-time Sign Language Recognition Script
## webcam_inference_osl.py

Perform live inference from webcam using your trained Uni-Sign ISLR model.

---

## Quick Start

```bash
python webcam_inference_osl.py
```

---

## System Requirements

### Python Environment
- Python 3.8+
- CUDA 11.8+ (recommended for GPU acceleration)

### Dependencies
```
torch>=2.0.0
torchvision
opencv-python>=4.5.0
mediapipe>=0.10.0
numpy>=1.20.0
```

### Hardware
- Webcam (built-in or USB)
- GPU (recommended) or CPU (slower)

---

## Installation

### 1. Navigate to the script directory
```bash
cd C:\Users\MOBPC\Downloads\FYP\FYPproject\OSL_Dataset\OSL_FineTuning
```

### 2. Install dependencies
```bash
pip install opencv-python mediapipe torch torchvision numpy
```

Or if you're using conda:
```bash
conda install -c pytorch pytorch::pytorch pytorch::pytorch-cuda=11.8 -c pytorch
conda install -c conda-forge opencv mediapipe numpy
```

### 3. Verify your trained checkpoint exists
```bash
# The script looks for:
# C:\Users\MOBPC\Downloads\FYP\FYPproject\Uni-Sign-main\Uni-Sign-main\out\osl_words_aug_only_finetuning\best_checkpoint.pth
```

---

## Usage

### Basic Usage
```bash
python webcam_inference_osl.py
```

### Specify Device (CPU or GPU)
```bash
# Use GPU (default if available)
python webcam_inference_osl.py --device cuda

# Use CPU only
python webcam_inference_osl.py --device cpu

# Auto-detect (default)
python webcam_inference_osl.py --device auto
```

---

## Keyboard Controls

| Key | Action |
|-----|--------|
| **SPACE** | Start/stop recording a sign sequence |
| **R** | Run recognition on recorded sequence (requires ≥5 frames) |
| **C** | Clear current sequence and restart |
| **Q** | Quit application |

---

## Workflow Example

1. **Start the script**
   ```bash
   python webcam_inference_osl.py
   ```

2. **Perform a sign**
   - Press SPACE to start recording
   - Perform your sign in front of the webcam
   - Press SPACE again to stop recording

3. **Get prediction**
   - Press R to run recognition
   - Top-5 predictions appear in console with confidence scores

4. **Repeat or exit**
   - Press C to clear and try another sign
   - Press Q to quit

---

## Screen Display

While running, the webcam window shows:

```
┌─────────────────────────────────────────────────┐
│ Recording: 45 frames                            │
│ Last: hello (85%)                   Frames: 892 │
│                                                   │
│ [Webcam feed with pose skeleton overlay]         │
│                                                   │
│ SPACE=record | R=recognize | C=clear | Q=quit   │
└─────────────────────────────────────────────────┘
```

Features:
- **Status bar**: Shows if recording and number of frames captured
- **Last prediction**: Displays the most recent recognition result
- **Pose skeleton**: MediaPipe overlays hand and body keypoints
- **Instructions**: Keyboard controls reminder

---

## Output Example

```
🔍 Recognizing 52 frames...

✅ Top-5 Predictions:
   1. hello                (88.4%)
   2. hi                   (7.2%)
   3. good                 (2.8%)
   4. morning              (1.3%)
   5. thanks               (0.3%)
```

---

## Features

### ✅ Real-time Inference
- Captures webcam frames at 30 FPS
- Extracts pose keypoints using MediaPipe
- Runs model inference on GPU/CPU

### ✅ Sequence Handling
- Maintains sliding window of last 64 frames
- Automatically pads/trims sequences to model input size
- Requires minimum 5 frames for recognition

### ✅ Pose Extraction
- 33 body landmarks (shoulders, elbows, wrists, hips, knees, ankles, head)
- 21 left hand keypoints (fingers and palm)
- 21 right hand keypoints (fingers and palm)
- Total: 261 features (3 coordinates per landmark)

### ✅ Visualization
- Live webcam feed with pose skeleton overlay
- Real-time status indicators
- FPS counter

### ✅ Prediction Smoothing
- Maintained history of last 3 predictions
- Helps identify consistent recognition across frames

---

## Troubleshooting

### "Cannot open webcam" error
**Cause**: Webcam not accessible or not connected
**Solution**:
1. Check if webcam is connected
2. Check if another application is using the webcam
3. Try unplugging/replugging the webcam
4. Check Windows Device Manager for camera device

### "Checkpoint not found" error
**Cause**: Model checkpoint doesn't exist at expected path
**Solution**:
1. Verify training completed successfully
2. Check path: `Uni-Sign-main/out/osl_words_aug_only_finetuning/best_checkpoint.pth`
3. Run training if not done yet

### "ModuleNotFoundError: No module named 'models.registry'"
**Cause**: Uni-Sign modules not in Python path
**Solution**:
1. Verify you're in the correct directory
2. Check Uni-Sign installation
3. Try: `pip install -e .` in Uni-Sign-main directory

### "MediaPipe not installed"
**Cause**: Missing dependency
**Solution**:
```bash
pip install mediapipe
```

### Low accuracy predictions
**Cause**: Poor lighting, unclear hand signs, or motion blur
**Solution**:
1. Ensure good lighting on your hands and body
2. Make clear, deliberate hand movements
3. Keep entire body in frame
4. Higher camera resolution if possible
5. Try multiple times - some variations work better

### GPU memory error
**Cause**: Running out of VRAM
**Solution**:
```bash
# Use CPU instead
python webcam_inference_osl.py --device cpu
```

---

## Configuration

Edit the `Config` class in `webcam_inference_osl.py` to customize:

```python
class Config:
    # Model paths
    PROJECT_ROOT = Path(r"C:\path\to\project")  # Change if needed
    UNISIGN_DIR = PROJECT_ROOT / "Uni-Sign-main" / "Uni-Sign-main"
    CHECKPOINT_PATH = UNISIGN_DIR / "out" / "osl_words_aug_only_finetuning" / "best_checkpoint.pth"
    
    # Model parameters
    MAX_SEQUENCE_LENGTH = 64       # Sequence length
    MIN_FRAMES_TO_RECOGNIZE = 5    # Minimum frames before recognition
    
    # Inference
    SMOOTHING_WINDOW = 3           # Number of past predictions to track
    CONFIDENCE_THRESHOLD = 0.3     # Minimum confidence to trust prediction
    
    # Webcam
    WEBCAM_WIDTH = 640
    WEBCAM_HEIGHT = 480
    WEBCAM_FPS = 30
    
    # MediaPipe sensitivity
    MP_DETECTION_CONFIDENCE = 0.5   # Higher = more strict pose detection
    MP_TRACKING_CONFIDENCE = 0.5    # Higher = more stable tracking
```

---

## File Structure

```
OSL_Dataset/
├── OSL_FineTuning/
│   ├── webcam_inference_osl.py          ← Main script
│   ├── finetuning_aug.ipynb             ← Training notebook
│   └── README_INFERENCE.md              ← This file
├── Words.txt                            ← Word labels (1 = hello, 2 = hi, ...)
└── ...

Uni-Sign-main/
└── Uni-Sign-main/
    ├── fine_tuning.py
    ├── models/
    │   └── registry.py
    ├── datasets.py
    └── out/
        └── osl_words_aug_only_finetuning/
            ├── best_checkpoint.pth      ← Model checkpoint
            ├── log.txt
            └── ...
```

---

## Performance Tips

### For Better Accuracy
1. **Good lighting**: Ensure hands and face are well-lit
2. **Clear background**: Avoid complex backgrounds
3. **Stable position**: Keep camera steady
4. **Full body**: Include torso in frame
5. **Natural movement**: Don't overly exaggerate signs

### For Better Speed
1. **Use GPU**: Much faster than CPU
2. **Lower webcam resolution**: Slightly impacts pose detection quality
3. **Reduce `SMOOTHING_WINDOW`**: Fewer past predictions to track

### For Better Stability
1. **Increase `MP_TRACKING_CONFIDENCE`**: More stable pose tracking
2. **Increase `MIN_FRAMES_TO_RECOGNIZE`**: More frames = more reliable
3. **Use `SMOOTHING_WINDOW`**: Majority vote smooths jitter

---

## Model Information

### Architecture
- **Task**: ISLR (Isolated Sign Language Recognition)
- **Model**: Uni-Sign (Vision Transformer + Pose Encoder)
- **Input**: Sequence of pose keypoint features (64 frames × 261 features)
- **Output**: Classification into word labels

### Training Details
- **Training Data**: Orange Sign Language (OSL) - Words subset
- **Augmentation**: Data augmentation applied during training
- **Checkpoint**: Best model saved after fine-tuning
- **Sequence Length**: 64 frames (at 30 FPS ≈ 2.1 seconds)

---

## Paper References

This script uses the Uni-Sign model. If you use this in research, please cite:

```bibtex
@article{unisign,
  title={Towards Unified Sign Language Recognition},
  author={...},
  journal={...},
  year={2023}
}
```

---

## FAQ

**Q: How many frames do I need?**
A: Minimum 5 frames (≈0.17 seconds), but 30-60 frames (1-2 seconds) gives better accuracy.

**Q: Does it recognize multiple signs in sequence?**
A: No, this is isolated sign recognition. Press C between signs to clear the buffer.

**Q: Can I use different hand signs than I trained on?**
A: Only if they're in your Words.txt label list. The model only recognizes words it was trained on.

**Q: How accurate is it?**
A: Accuracy depends on training quality. Check your training logs for validation accuracy.
At 64 frames, expect 70-95% top-1 accuracy on test set (varies by model).

**Q: Can I save predictions?**
A: Modify the script to log predictions. See `_handle_recognition()` method.

---

## Advanced: Running Without Webcam (Batch Inference)

To run inference on saved videos instead of webcam, modify the script to load from file path. See the commented section in `WebcamApp.run()`.

---

## Support & Issues

If you encounter issues:
1. Check the Troubleshooting section above
2. Verify configuration paths exist
3. Check console output for specific error messages
4. Ensure all dependencies are installed
5. Try CPU mode if GPU fails

---

**Last Updated**: March 2026  
**Status**: Ready for inference
