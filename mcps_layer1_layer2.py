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

# ==========================================
# LAYER 2: METAHEURISTIC OPTIMIZATION
# ==========================================
class ExpertQuartetOptimizer:
    def __init__(self, feature_vector, num_agents=30, iterations=50):
        self.features = np.array(feature_vector)
        self.num_agents = num_agents
        self.iterations = iterations
        self.dimensions = len(self.features)
        
        print(f"\n[Layer 2] Initialized with Extracted Biometric Vector: {self.features}")

    def fitness_function(self, weights):
        # Placeholder objective function for feature weighting
        weighted_features = self.features * weights
        return np.sum(weighted_features) 

    def run_pso(self):
        print("--- Executing Particle Swarm Optimization (PSO) ---")
        w, c1, c2 = 0.5, 1.5, 1.5 
        
        positions = np.random.uniform(0, 1, (self.num_agents, self.dimensions))
        velocities = np.zeros((self.num_agents, self.dimensions))
        
        pbest = positions.copy()
        pbest_scores = np.array([self.fitness_function(p) for p in positions])
        
        gbest = pbest[np.argmax(pbest_scores)]
        
        for t in range(self.iterations):
            for i in range(self.num_agents):
                r1, r2 = np.random.rand(), np.random.rand()
                
                velocities[i] = (w * velocities[i] + 
                                 c1 * r1 * (pbest[i] - positions[i]) + 
                                 c2 * r2 * (gbest - positions[i]))
                
                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], 0, 1) 
                
                current_score = self.fitness_function(positions[i])
                if current_score > pbest_scores[i]:
                    pbest[i] = positions[i]
                    pbest_scores[i] = current_score
                    
            gbest = pbest[np.argmax(pbest_scores)]

        print(f"PSO Global Best Weights: {gbest}")
        return gbest

    def run_firefly_algorithm(self):
        print("\n--- Executing Firefly Algorithm (FA) ---")
        alpha, beta0, gamma = 0.2, 1.0, 1.0 
        
        fireflies = np.random.uniform(0, 1, (self.num_agents, self.dimensions))
        light_intensity = np.array([self.fitness_function(f) for f in fireflies])

        for t in range(self.iterations):
            for i in range(self.num_agents):
                for j in range(self.num_agents):
                    if light_intensity[j] > light_intensity[i]:
                        r = np.linalg.norm(fireflies[i] - fireflies[j])
                        beta = beta0 * np.exp(-gamma * r**2)
                        
                        random_step = alpha * (np.random.rand(self.dimensions) - 0.5)
                        fireflies[i] += beta * (fireflies[j] - fireflies[i]) + random_step
                        fireflies[i] = np.clip(fireflies[i], 0, 1)
                        
                        light_intensity[i] = self.fitness_function(fireflies[i])
        
        best_firefly = fireflies[np.argmax(light_intensity)]
        print(f"FA Best Firefly Weights: {best_firefly}")
        return best_firefly

# ==========================================
# LAYER 1: PERCEPTION & KINEMATICS
# ==========================================
def process_live_camera(pose_model='pose_landmarker.task', face_model='face_landmarker.task'):
    print("[Layer 1] Initializing Dual-Model MCPS Pipeline (Pose + Face Mesh)...")
    
    pose_base = python.BaseOptions(model_asset_path=pose_model)
    pose_options = vision.PoseLandmarkerOptions(
        base_options=pose_base, running_mode=vision.RunningMode.VIDEO, min_pose_detection_confidence=0.5
    )
    
    face_base = python.BaseOptions(model_asset_path=face_model)
    face_options = vision.FaceLandmarkerOptions(
        base_options=face_base, running_mode=vision.RunningMode.VIDEO, min_face_detection_confidence=0.5
    )
    
    movement_signals, eye_aperture_signals = [], []
    prev_landmarks = None
    
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
            
            timestamp_ms = int((time.perf_counter() - start_time) * 1000)
            
            pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)
            face_result = face_landmarker.detect_for_video(mp_image, timestamp_ms)
            
            height, width, _ = frame.shape

            if pose_result.pose_landmarks:
                p_marks = pose_result.pose_landmarks[0]
                l_shoulder = np.array([p_marks[11].x, p_marks[11].y, p_marks[11].z])
                r_shoulder = np.array([p_marks[12].x, p_marks[12].y, p_marks[12].z])
                shoulder_distance = np.linalg.norm(l_shoulder - r_shoulder) + 1e-6
                
                target_nodes = [0, 11, 12, 13, 14, 15, 16]
                current_coords = np.array([[p_marks[i].x, p_marks[i].y, p_marks[i].z] for i in target_nodes])
                
                if prev_landmarks is not None:
                    raw_displacement = np.linalg.norm(current_coords - prev_landmarks)
                    movement_signals.append(raw_displacement / shoulder_distance)
                prev_landmarks = current_coords

                lx, ly = int(p_marks[11].x * width), int(p_marks[11].y * height)
                rx, ry = int(p_marks[12].x * width), int(p_marks[12].y * height)
                cv2.line(frame, (lx, ly), (rx, ry), (255, 0, 255), 3)
                
                for idx in target_nodes:
                    cx, cy = int(p_marks[idx].x * width), int(p_marks[idx].y * height)
                    cv2.circle(frame, (cx, cy), 6, (0, 255, 255), -1) 
                    cv2.circle(frame, (cx, cy), 6, (0, 0, 0), 1)

            if face_result.face_landmarks:
                f_marks = face_result.face_landmarks[0]
                
                for idx in [468, 473]:
                    cv2.circle(frame, (int(f_marks[idx].x * width), int(f_marks[idx].y * height)), 3, (0, 0, 255), -1)
                for idx in [159, 145, 386, 374]:
                    cv2.circle(frame, (int(f_marks[idx].x * width), int(f_marks[idx].y * height)), 2, (255, 0, 0), -1)

                r_eye_aperture = np.linalg.norm(
                    np.array([f_marks[159].x, f_marks[159].y]) - np.array([f_marks[145].x, f_marks[145].y])
                )
                eye_aperture_signals.append(r_eye_aperture)

            cv2.putText(frame, "MCPS Dual-Model Active (Press 'Q' to End & Optimize)", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('ADHD Real-Time Monitoring', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
        
    m_array = np.array(movement_signals) if movement_signals else np.array([0.0])
    e_array = np.array(eye_aperture_signals) if eye_aperture_signals else np.array([0.0])
    
    print("\n[Layer 1] Session Complete. Compiling Biometric Vector...")
    return {
        "mean_displacement": float(np.mean(m_array)),
        "fidget_frequency_threshold": float(np.sum(m_array > 0.05) / len(m_array)) if len(m_array) > 0 else 0.0,
        "mean_eye_aperture": float(np.mean(e_array)),
        "eye_aperture_variance": float(np.var(e_array))
    }

# ==========================================
# SYSTEM EXECUTION
# ==========================================
if __name__ == "__main__":
    print("=== STARTING MEDICAL CYBER-PHYSICAL SYSTEM ===")
    
    # 1. Execute Layer 1 and hold the dictionary output
    metrics = process_live_camera()
    
    # 2. Dynamically extract the values into a mathematical vector
    extracted_vector = [
        metrics["mean_displacement"],
        metrics["fidget_frequency_threshold"],
        metrics["mean_eye_aperture"],
        metrics["eye_aperture_variance"]
    ]
    
    # 3. Automatically trigger Layer 2 using the extracted data
    layer2 = ExpertQuartetOptimizer(feature_vector=extracted_vector, num_agents=20, iterations=30)
    
    pso_optimal_weights = layer2.run_pso()
    fa_optimal_weights = layer2.run_firefly_algorithm()
    
    print("\n[System] Layers 1 & 2 Execution Complete. Ready for Layers 3 & 4.")