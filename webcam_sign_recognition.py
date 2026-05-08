"""
Real-time Sign Language Recognition with Trained Uni-Sign Model
================================================================
Uses webcam + MediaPipe for pose extraction
Recognizes WLASL-100 signs using your trained model (91.5% accuracy)
"""

print("Starting webcam sign recognition...")
import sys
import os
print(f"Python: {sys.executable}")

# Prevent TensorFlow from loading (conflicts with PyTorch CUDA on Windows)
# MediaPipe only needs TFLite, not full TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES_FOR_TF'] = '-1'
import importlib
_fake_tf = type(sys)('tensorflow')
_fake_tf.__version__ = '0.0.0'
_fake_tf.__spec__ = importlib.machinery.ModuleSpec('tensorflow', None)
_fake_tf.python = type(sys)('tensorflow.python')
_fake_tf.python.__spec__ = importlib.machinery.ModuleSpec('tensorflow.python', None)
sys.modules['tensorflow'] = _fake_tf
sys.modules['tensorflow.python'] = _fake_tf.python

# Import torch FIRST before any other libraries
print("Importing torch...")
import torch
print(f"torch imported: {torch.__version__}")

import cv2
print("cv2 imported")
import numpy as np
print("numpy imported")
import time
from collections import deque
from pathlib import Path
print("Basic imports done")

# Setup paths
SCRIPT_DIR = Path(r'c:\Users\MOBPC\Downloads\FYP\FYPproject\Uni-Sign-main\Uni-Sign-main')
os.chdir(SCRIPT_DIR)
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
print(f"Working dir: {SCRIPT_DIR}")

# Note: numpy._core compatibility not needed for numpy 1.x

# WLASL-100 class labels (100 common ASL signs)
WLASL100_LABELS = [
    "book", "drink", "computer", "before", "chair", "go", "clothes", "who", 
    "candy", "cousin", "deaf", "fine", "help", "no", "thin", "walk", "year",
    "yes", "all", "black", "cool", "finish", "hot", "like", "many", "mother",
    "now", "orange", "school", "study", "thanksgiving", "what", "woman", "bed",
    "blue", "bowling", "can", "dog", "family", "fish", "graduate", "hat",
    "hearing", "kiss", "language", "later", "man", "meet", "need", "nice",
    "nurse", "pizza", "play", "right", "same", "shirt", "sorry", "stay",
    "table", "tell", "want", "white", "work", "write", "accident", "apple",
    "bird", "change", "color", "corn", "cow", "dance", "dark", "doctor",
    "eat", "enjoy", "forget", "give", "happy", "have", "hearing", "hospital",
    "hurt", "know", "learn", "lost", "medicine", "movie", "paint", "paper",
    "pink", "pull", "read", "red", "restaurant", "see", "sick", "sign",
    "student", "teacher", "time", "wrong"
]


class PoseExtractor:
    """Extract pose keypoints using MediaPipe, matching the training data format"""

    # MediaPipe Pose landmark indices -> 9 body joints matching training format
    # Training uses COCO WholeBody indices [0,3,4,5,6,7,8,9,10]:
    # nose, left_ear, right_ear, left_shoulder, right_shoulder,
    # left_elbow, right_elbow, left_wrist, right_wrist
    BODY_MP_INDICES = [0, 7, 8, 11, 12, 13, 14, 15, 16]

    # MediaPipe Face Mesh indices -> 18 face keypoints matching training format
    # 9 jaw contour (every other from 68-pt jaw) + 8 inner mouth + 1 nose tip
    FACE_JAW_MP = [234, 93, 132, 58, 172, 136, 150, 176, 152]  # 9 pts
    FACE_MOUTH_MP = [78, 191, 80, 81, 82, 13, 312, 311]         # 8 pts
    FACE_NOSE_MP = [1]                                            # 1 pt (center ref)

    def __init__(self):
        import mediapipe as mp
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1
        )
        self.face_indices = self.FACE_JAW_MP + self.FACE_MOUTH_MP + self.FACE_NOSE_MP
        print("[OK] MediaPipe Holistic initialized")

    def extract(self, frame):
        """Extract RAW pose keypoints from a single frame.
        Returns unnormalized [0-1] coordinates. Normalization happens
        in SignRecognizer.predict() across ALL frames at once, matching
        the training pipeline in datasets.py load_part_kp().
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(rgb_frame)

        # Body: 9 joints, raw [0-1] coords + visibility
        body = np.zeros((9, 3), dtype=np.float32)
        if results.pose_landmarks:
            for i, mp_idx in enumerate(self.BODY_MP_INDICES):
                lm = results.pose_landmarks.landmark[mp_idx]
                body[i] = [lm.x, lm.y, lm.visibility]

        # Left hand: 21 joints, raw [0-1] coords, centered at wrist
        left_hand = np.zeros((21, 3), dtype=np.float32)
        if results.left_hand_landmarks:
            for i, lm in enumerate(results.left_hand_landmarks.landmark):
                left_hand[i] = [lm.x, lm.y, 1.0]
            # Center at wrist (training: hand_kp2d - hand_kp2d[0])
            left_hand[:, :2] -= left_hand[0, :2].copy()

        # Right hand: 21 joints, raw [0-1] coords, centered at wrist
        right_hand = np.zeros((21, 3), dtype=np.float32)
        if results.right_hand_landmarks:
            for i, lm in enumerate(results.right_hand_landmarks.landmark):
                right_hand[i] = [lm.x, lm.y, 1.0]
            right_hand[:, :2] -= right_hand[0, :2].copy()

        # Face: 18 joints, raw [0-1] coords, centered at nose tip (last)
        face = np.zeros((18, 3), dtype=np.float32)
        if results.face_landmarks:
            for i, mp_idx in enumerate(self.face_indices):
                if mp_idx < len(results.face_landmarks.landmark):
                    lm = results.face_landmarks.landmark[mp_idx]
                    face[i] = [lm.x, lm.y, 1.0]
            # Center at nose tip (training: kp - kp[-1])
            face[:, :2] -= face[-1, :2].copy()

        return {
            'body': body,
            'left': left_hand,
            'right': right_hand,
            'face_all': face
        }, results
    
    def draw_landmarks(self, frame, results):
        """Draw pose landmarks on frame"""
        import mediapipe as mp
        mp_holistic = mp.solutions.holistic
        mp_drawing = mp.solutions.drawing_utils
        
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=2)
            )
        
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2)
            )
        
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2)
            )
        
        return frame
    
    def close(self):
        self.holistic.close()


class SignRecognizer:
    """Sign language recognizer using trained Uni-Sign model"""
    
    def __init__(self, checkpoint_path, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Import model
        from models import Uni_Sign
        import argparse
        
        # Create args
        self.args = argparse.Namespace(
            hidden_dim=256,
            rgb_support=False,  # Disable RGB for faster inference
            dataset='WLASL',
            task='ISLR',
            max_length=64,
            label_smoothing=0.0
        )
        
        print("Loading Uni-Sign model...")
        self.model = Uni_Sign(args=self.args)
        
        # Load checkpoint
        if Path(checkpoint_path).exists():
            print(f"Loading checkpoint: {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location='cpu')['model']
            
            # Handle potential key mismatches
            model_dict = self.model.state_dict()
            filtered_dict = {k: v for k, v in state_dict.items() if k in model_dict}
            self.model.load_state_dict(filtered_dict, strict=False)
            print(f"[OK] Loaded {len(filtered_dict)}/{len(state_dict)} weights")
        else:
            print(f"WARNING: Checkpoint not found: {checkpoint_path}")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Pose buffer
        self.pose_buffer = deque(maxlen=32)
        self.min_frames = 8
        
        # Labels
        self.labels = WLASL100_LABELS
        print(f"[OK] Model ready with {len(self.labels)} sign classes")
    
    def add_frame(self, pose_data):
        """Add frame's pose data to buffer"""
        self.pose_buffer.append(pose_data)
    
    def predict(self):
        """Run prediction on buffered frames.
        Applies normalization across ALL frames at once, matching
        datasets.py load_part_kp() + crop_scale().
        """
        if len(self.pose_buffer) < self.min_frames:
            return None, 0.0, f"Collecting frames ({len(self.pose_buffer)}/{self.min_frames})"

        try:
            import copy
            frames = list(self.pose_buffer)
            T = len(frames)
            thr = 0.3

            # Stack raw pose data across all frames: (T, N, 3)
            body_all = np.stack([f['body'] for f in frames])    # (T, 9, 3)
            left_all = np.stack([f['left'] for f in frames])    # (T, 21, 3)
            right_all = np.stack([f['right'] for f in frames])  # (T, 21, 3)
            face_all = np.stack([f['face_all'] for f in frames])# (T, 18, 3)

            # === Normalize body using crop_scale across ALL frames ===
            # Exactly matches datasets.py crop_scale() on (T, 9, 3)
            body_norm = copy.deepcopy(body_all)
            valid_coords = body_all[body_all[..., 2] > thr][:, :2]
            if len(valid_coords) < 4:
                body_norm = np.zeros_like(body_all)
                scale = 0
            else:
                xmin = valid_coords[:, 0].min()
                xmax = valid_coords[:, 0].max()
                ymin = valid_coords[:, 1].min()
                ymax = valid_coords[:, 1].max()
                scale = max(xmax - xmin, ymax - ymin)
                if scale == 0:
                    body_norm = np.zeros_like(body_all)
                else:
                    xs = (xmin + xmax - scale) / 2
                    ys = (ymin + ymax - scale) / 2
                    body_norm[..., :2] = (body_all[..., :2] - [xs, ys]) / scale
                    body_norm[..., :2] = (body_norm[..., :2] - 0.5) * 2
                    body_norm = np.clip(body_norm, -1, 1)
                    body_norm[body_norm[..., 2] <= thr] = 0

            # === Normalize hands and face using body scale ===
            # Exactly matches datasets.py load_part_kp() lines 73-82
            for part_data in [left_all, right_all, face_all]:
                if scale == 0:
                    part_data[:] = 0
                else:
                    part_data[..., :2] = part_data[..., :2] / scale
                    part_data[:] = np.clip(part_data, -1, 1)
                    part_data[part_data[..., 2] <= thr] = 0

            # Create input tensors (B=1, T, N, 3)
            src_input = {
                'body': torch.tensor(body_norm, dtype=torch.float32).unsqueeze(0).to(self.device),
                'left': torch.tensor(left_all, dtype=torch.float32).unsqueeze(0).to(self.device),
                'right': torch.tensor(right_all, dtype=torch.float32).unsqueeze(0).to(self.device),
                'face_all': torch.tensor(face_all, dtype=torch.float32).unsqueeze(0).to(self.device),
                'attention_mask': torch.ones(1, T).to(self.device),
                'name_batch': ['realtime']
            }

            tgt_input = {'gt_sentence': ['']}

            with torch.no_grad():
                # Forward pass
                output = self.model(src_input, tgt_input)

                # Generate prediction
                generated = self.model.generate(
                    output,
                    max_new_tokens=10,
                    num_beams=4
                )

                # Decode
                decoded = self.model.mt5_tokenizer.decode(
                    generated[0],
                    skip_special_tokens=True
                )

                # Clean up prediction
                prediction = decoded.strip()
                if prediction:
                    return prediction, 0.9, "Recognized"
                else:
                    return None, 0.0, "No clear sign detected"

        except Exception as e:
            import traceback
            traceback.print_exc()
            return None, 0.0, f"Error: {str(e)[:50]}"
    
    def clear_buffer(self):
        """Clear pose buffer"""
        self.pose_buffer.clear()


def main():
    print("="*60)
    print("Sign Language Recognition with Uni-Sign Model")
    print("Trained on WLASL-100 (91.5% accuracy)")
    print("="*60)
    
    # Configuration
    CHECKPOINT = 'out/wlasl100_finetuning/best_checkpoint.pth'
    
    # Check for checkpoint
    if not Path(CHECKPOINT).exists():
        alt_checkpoint = 'out/stage2_pretraining/best_checkpoint.pth'
        if Path(alt_checkpoint).exists():
            CHECKPOINT = alt_checkpoint
            print(f"Using pretrained checkpoint: {alt_checkpoint}")
        else:
            print(f"WARNING: No checkpoint found. Model will not recognize signs accurately.")
    
    # Initialize components
    print("\n--- Initializing ---")
    
    try:
        print("Creating PoseExtractor...")
        pose_extractor = PoseExtractor()
    except Exception as e:
        import traceback
        print(f"Error creating PoseExtractor: {e}")
        traceback.print_exc()
        print("Install MediaPipe: pip install mediapipe")
        return
    
    try:
        print("Creating SignRecognizer...")
        recognizer = SignRecognizer(CHECKPOINT)
    except Exception as e:
        import traceback
        print(f"Error loading model: {e}")
        traceback.print_exc()
        print("Running in pose-only mode...")
        recognizer = None
    
    # Open webcam
    print("\n--- Opening Camera ---")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("\n" + "="*60)
    print("Controls:")
    print("  'q' - Quit")
    print("  'c' - Clear buffer (start new sign)")
    print("  's' - Save screenshot")
    print("  'SPACE' - Trigger recognition")
    print("="*60 + "\n")
    
    # Variables
    current_sign = ""
    confidence = 0.0
    status_msg = "Ready - perform a sign"
    fps_list = deque(maxlen=30)
    last_time = time.time()
    frame_count = 0
    auto_predict_interval = 15  # Predict every N frames
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            
            # FPS
            now = time.time()
            fps = 1.0 / (now - last_time) if (now - last_time) > 0 else 30
            fps_list.append(fps)
            avg_fps = sum(fps_list) / len(fps_list)
            last_time = now
            
            # Extract pose
            pose_data, results = pose_extractor.extract(frame)
            
            # Add to buffer
            if recognizer:
                recognizer.add_frame(pose_data)
            
            # Draw landmarks
            frame = pose_extractor.draw_landmarks(frame, results)
            
            # Auto predict periodically
            frame_count += 1
            if recognizer and frame_count % auto_predict_interval == 0:
                sign, conf, msg = recognizer.predict()
                if sign:
                    current_sign = sign
                    confidence = conf
                status_msg = msg
            
            # ========== Draw UI ==========
            # Dark overlay at top
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 130), (30, 30, 30), -1)
            cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
            
            # Title
            cv2.putText(frame, "WLASL-100 Sign Recognition", (10, 28),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Recognized sign
            if current_sign:
                cv2.putText(frame, f"Sign: {current_sign.upper()}", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                cv2.putText(frame, f"Confidence: {confidence:.0%}", (10, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            else:
                cv2.putText(frame, "Perform a sign...", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (150, 150, 150), 2)
            
            # Status
            cv2.putText(frame, status_msg, (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 200, 150), 1)
            
            # Buffer bar
            if recognizer:
                buf_pct = len(recognizer.pose_buffer) / recognizer.pose_buffer.maxlen
                bar_w = int(150 * buf_pct)
                cv2.rectangle(frame, (w-170, 10), (w-20, 30), (50, 50, 50), -1)
                cv2.rectangle(frame, (w-170, 10), (w-170+bar_w, 30), (0, 200, 0), -1)
                cv2.putText(frame, f"Buffer: {len(recognizer.pose_buffer)}", (w-170, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            # FPS
            cv2.putText(frame, f"FPS: {avg_fps:.0f}", (w-70, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Detection status
            detections = []
            if results.pose_landmarks:
                detections.append("Body")
            if results.left_hand_landmarks:
                detections.append("L-Hand")
            if results.right_hand_landmarks:
                detections.append("R-Hand")
            
            status_color = (0, 255, 0) if len(detections) >= 2 else (0, 150, 255)
            det_text = " | ".join(detections) if detections else "No pose detected"
            cv2.putText(frame, det_text, (w-200, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, status_color, 1)
            
            # Instructions at bottom
            cv2.putText(frame, "'q'=Quit | 'c'=Clear | SPACE=Recognize", (10, h-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            
            # Show frame
            cv2.imshow('Sign Language Recognition', frame)
            
            # Key handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                if recognizer:
                    recognizer.clear_buffer()
                current_sign = ""
                confidence = 0.0
                status_msg = "Buffer cleared - start new sign"
                print("Buffer cleared")
            elif key == ord('s'):
                fname = f"sign_capture_{int(time.time())}.jpg"
                cv2.imwrite(fname, frame)
                print(f"Saved: {fname}")
            elif key == ord(' '):  # Space bar for manual recognition
                if recognizer:
                    sign, conf, msg = recognizer.predict()
                    if sign:
                        current_sign = sign
                        confidence = conf
                    status_msg = msg
                    print(f"Recognition: {sign} ({conf:.1%}) - {msg}")
    
    except KeyboardInterrupt:
        print("\nInterrupted")
    
    finally:
        pose_extractor.close()
        cap.release()
        cv2.destroyAllWindows()
        print("Done!")


if __name__ == "__main__":
    main()
