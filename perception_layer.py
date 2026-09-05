import os

# 1. Suppress TensorFlow C++ logging (0 = ALL, 1 = filter INFO, 2 = filter WARNING, 3 = filter ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 2. Suppress MediaPipe GLOG outputs
os.environ['GLOG_minloglevel'] = '3'

# 3. Disable Google Clearcut telemetry uploader
os.environ['MEDIAPIPE_DISABLE_TELEMETRY'] = '1'

import cv2
import numpy as np
import mediapipe as mp
# ... rest of your imports and code

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def process_patient_video(video_path, model_path='pose_landmarker.task'):
    print(f"[Layer 1] Initializing MediaPipe Tasks API for: {video_path}")
    
    # Configure the modern Tasks API for continuous video streams
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    movement_signals = []
    prev_landmarks = None
    
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_index = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # MediaPipe strictly requires RGB formatting
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            
            # Calculate precise millisecond timestamp for the Tasks API
            timestamp_ms = int((frame_index / fps) * 1000)
            pose_result = landmarker.detect_for_video(mp_image, timestamp_ms)
            
            if pose_result.pose_landmarks:
                landmarks = pose_result.pose_landmarks[0]
                
                # Extract shoulders for Skeletal Anchor Scale Normalization
                l_shoulder = np.array([landmarks[11].x, landmarks[11].y, landmarks[11].z])
                r_shoulder = np.array([landmarks[12].x, landmarks[12].y, landmarks[12].z])
                shoulder_distance = np.linalg.norm(l_shoulder - r_shoulder) + 1e-6
                
                # Target upper-body kinetic points: Nose, Shoulders, Elbows, Wrists
                target_nodes = [0, 11, 12, 13, 14, 15, 16]
                current_coords = np.array([[landmarks[i].x, landmarks[i].y, landmarks[i].z] for i in target_nodes])
                
                if prev_landmarks is not None:
                    raw_displacement = np.linalg.norm(current_coords - prev_landmarks)
                    normalized_displacement = raw_displacement / shoulder_distance
                    movement_signals.append(normalized_displacement)
                    
                prev_landmarks = current_coords
            frame_index += 1
            
        cap.release()
        
    movement_array = np.array(movement_signals) if movement_signals else np.array([0.0])
    
    # Map the localized video physics into structural statistics
    return {
        "mean_displacement": float(np.mean(movement_array)),
        "variance_displacement": float(np.var(movement_array)),
        "max_peak_movement": float(np.max(movement_array)),
        "fidget_frequency_threshold": float(np.sum(movement_array > 0.05) / len(movement_array))
    }

if __name__ == "__main__":
    # Test the script on one of your video files
    metrics = process_patient_video("test_video_1.mp4")
    print(metrics)
    pass