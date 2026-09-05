import os
import time

# Suppress C++ telemetry and logging noise
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '3'
os.environ['MEDIAPIPE_DISABLE_TELEMETRY'] = '1'

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def process_live_camera(pose_model='pose_landmarker.task', face_model='face_landmarker.task'):
    print("[Layer 1] Initializing Dual-Model MCPS Pipeline (Pose + Face Mesh)...")
    
    # --- MODEL CONFIGURATIONS ---
    pose_base = python.BaseOptions(model_asset_path=pose_model)
    pose_options = vision.PoseLandmarkerOptions(
        base_options=pose_base, 
        running_mode=vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.5
    )
    
    face_base = python.BaseOptions(model_asset_path=face_model)
    face_options = vision.FaceLandmarkerOptions(
        base_options=face_base, 
        running_mode=vision.RunningMode.VIDEO,
        min_face_detection_confidence=0.5
    )
    
    movement_signals = []
    eye_aperture_signals = []
    prev_landmarks = None
    
    # Instantiate both models concurrently
    with vision.PoseLandmarker.create_from_options(pose_options) as pose_landmarker, \
         vision.FaceLandmarker.create_from_options(face_options) as face_landmarker:
        
        cap = cv2.VideoCapture(0)
        print("[System] Camera Active. Press 'q' in the video window to stop recording.")
        start_time = time.perf_counter()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            
            current_time = time.perf_counter()
            timestamp_ms = int((current_time - start_time) * 1000)
            
            # Execute parallel inferences
            pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)
            face_result = face_landmarker.detect_for_video(mp_image, timestamp_ms)
            
            height, width, _ = frame.shape

            # --- 1. KINEMATIC POSE EXTRACTION ---
            if pose_result.pose_landmarks:
                p_marks = pose_result.pose_landmarks[0]
                
                l_shoulder = np.array([p_marks[11].x, p_marks[11].y, p_marks[11].z])
                r_shoulder = np.array([p_marks[12].x, p_marks[12].y, p_marks[12].z])
                shoulder_distance = np.linalg.norm(l_shoulder - r_shoulder) + 1e-6
                
                target_nodes = [0, 11, 12, 13, 14, 15, 16]
                current_coords = np.array([[p_marks[i].x, p_marks[i].y, p_marks[i].z] for i in target_nodes])
                
                if prev_landmarks is not None:
                    raw_displacement = np.linalg.norm(current_coords - prev_landmarks)
                    normalized_displacement = raw_displacement / shoulder_distance
                    movement_signals.append(normalized_displacement)
                prev_landmarks = current_coords

                # Draw Pose Overlays
                lx, ly = int(p_marks[11].x * width), int(p_marks[11].y * height)
                rx, ry = int(p_marks[12].x * width), int(p_marks[12].y * height)
                cv2.line(frame, (lx, ly), (rx, ry), (255, 0, 255), 3)
                
                for idx in target_nodes:
                    cx, cy = int(p_marks[idx].x * width), int(p_marks[idx].y * height)
                    cv2.circle(frame, (cx, cy), 6, (0, 255, 255), -1) 
                    cv2.circle(frame, (cx, cy), 6, (0, 0, 0), 1)

            # --- 2. OCULOMOTOR FACE EXTRACTION ---
            if face_result.face_landmarks:
                f_marks = face_result.face_landmarks[0]
                
                pupils = [468, 473]
                eyelids = [159, 145, 386, 374]
                
                # Draw Pupils in Red
                for idx in pupils:
                    cx, cy = int(f_marks[idx].x * width), int(f_marks[idx].y * height)
                    cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
                    
                # Draw Eyelids in Blue
                for idx in eyelids:
                    cx, cy = int(f_marks[idx].x * width), int(f_marks[idx].y * height)
                    cv2.circle(frame, (cx, cy), 2, (255, 0, 0), -1)

                # Calculate Eye Aperture (Distance between top and bottom eyelid)
                r_eye_aperture = np.linalg.norm(
                    np.array([f_marks[159].x, f_marks[159].y]) - 
                    np.array([f_marks[145].x, f_marks[145].y])
                )
                eye_aperture_signals.append(r_eye_aperture)

            cv2.putText(frame, "Dual-Model Biomarker Tracking Active", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('ADHD Real-Time Monitoring', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
        
    m_array = np.array(movement_signals) if movement_signals else np.array([0.0])
    e_array = np.array(eye_aperture_signals) if eye_aperture_signals else np.array([0.0])
    
    print("\n[Layer 1] Session Complete. Generating Biometric Statistics...")
    return {
        "mean_displacement": float(np.mean(m_array)),
        "fidget_frequency_threshold": float(np.sum(m_array > 0.05) / len(m_array)) if len(m_array) > 0 else 0.0,
        "mean_eye_aperture (Blink Metric)": float(np.mean(e_array)),
        "eye_aperture_variance (Distraction Metric)": float(np.var(e_array))
    }

if __name__ == "__main__":
    metrics = process_live_camera()
    print("\n=== Final Session Metrics ===")
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")