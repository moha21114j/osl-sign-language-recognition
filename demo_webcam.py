"""
Simple Sign Language Webcam Demo
================================
Shows pose skeleton + basic gesture detection
Works without the full Uni-Sign model
"""

import cv2
import numpy as np
import time
from collections import deque
import mediapipe as mp

def main():
    print("="*50)
    print("Simple Sign Language Demo")
    print("="*50)
    
    # Initialize MediaPipe
    print("Initializing MediaPipe...")
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    
    holistic = mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    print("✓ MediaPipe ready")
    
    # Open webcam
    print("Opening camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("✓ Camera ready")
    
    print("\nControls: 'q' to quit, 's' to save frame")
    print("="*50 + "\n")
    
    fps_list = deque(maxlen=30)
    last_time = time.time()
    
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
            
            # Process with MediaPipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)
            
            # Draw landmarks
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
            
            # Simple gesture detection
            gesture = "Ready"
            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                l_wrist = lm[mp_holistic.PoseLandmark.LEFT_WRIST]
                r_wrist = lm[mp_holistic.PoseLandmark.RIGHT_WRIST]
                l_shoulder = lm[mp_holistic.PoseLandmark.LEFT_SHOULDER]
                r_shoulder = lm[mp_holistic.PoseLandmark.RIGHT_SHOULDER]
                
                l_up = l_wrist.y < l_shoulder.y and l_wrist.visibility > 0.5
                r_up = r_wrist.y < r_shoulder.y and r_wrist.visibility > 0.5
                
                if l_up and r_up:
                    gesture = "BOTH HANDS UP!"
                elif l_up:
                    gesture = "Left hand up"
                elif r_up:
                    gesture = "Right hand up"
            
            # Draw UI
            cv2.rectangle(frame, (0, 0), (w, 80), (30, 30, 30), -1)
            cv2.putText(frame, "Sign Language Demo", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"Gesture: {gesture}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"FPS: {avg_fps:.0f}", (w-80, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Status
            status = []
            if results.pose_landmarks: status.append("Body")
            if results.left_hand_landmarks: status.append("L-Hand")
            if results.right_hand_landmarks: status.append("R-Hand")
            cv2.putText(frame, " | ".join(status) if status else "No detection", 
                       (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            
            cv2.imshow('Sign Language Demo', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite(f"capture_{int(time.time())}.jpg", frame)
                print("Saved!")
    
    except KeyboardInterrupt:
        print("\nStopped")
    
    finally:
        holistic.close()
        cap.release()
        cv2.destroyAllWindows()
        print("Done!")

if __name__ == "__main__":
    main()
