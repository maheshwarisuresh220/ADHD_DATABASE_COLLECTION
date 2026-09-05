import os
import time
import random
import math
from collections import deque
import cv2
import numpy as np

# Suppress C++ telemetry and logging noise for professional demonstration
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '3'
os.environ['MEDIAPIPE_DISABLE_TELEMETRY'] = '1'

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ==========================================
# LAYER 2: METAHEURISTIC ADFI OPTIMIZATION
# ==========================================
class ADFI_Optimizer:
    def __init__(self, interval_data, num_agents=30, iterations=50):
        """
        Optimizes the exact alpha, beta, and w coefficients from the thesis equations.
        interval_data: List of dictionaries containing raw metrics per interval i.
        """
        self.interval_data = interval_data
        self.num_agents = num_agents
        self.iterations = iterations
        # 12 Dimensions: [alpha1..5, beta1..5, w1, w2]
        self.dimensions = 12 
        
        print(f"\n[Layer 2] Initialized with {len(self.interval_data)} Assessment Intervals.")

    def calculate_adfi(self, weights, data):
        alphas = weights[0:5]
        betas = weights[5:10]
        w1, w2 = weights[10], weights[11]
        
        S_beh = 0.0
        S_perf = 0.0
        
        for interval in data:
            # Equation 3.10: D_i
            D_i = (alphas[0] * interval['velocity'] + 
                   alphas[1] * interval['acceleration'] + 
                   alphas[2] * interval['variance'] + 
                   alphas[3] * interval['blinks'] + 
                   alphas[4] * interval['gaze_loss'])
            
            # Equation 3.11: I_j
            I_j = (betas[0] * interval['omissions'] + 
                   betas[1] * interval['commissions'] + 
                   betas[2] * interval['avg_rt'] + 
                   betas[3] * interval['var_rt'] + 
                   betas[4] * interval['timeouts'])
                   
            S_beh += D_i
            S_perf += I_j
            
        # Equation 3.12: Final ADFI
        ADFI = (w1 * S_beh) + (w2 * S_perf)
        return ADFI, S_beh, S_perf

    def fitness_function(self, weights):
        ADFI, _, _ = self.calculate_adfi(weights, self.interval_data)
        return ADFI 

    def run_pso(self):
        print("--- Executing PSO for ADFI Coefficient Optimization ---")
        w, c1, c2 = 0.5, 1.5, 1.5 
        positions = np.random.uniform(0, 1, (self.num_agents, self.dimensions))
        velocities = np.zeros((self.num_agents, self.dimensions))
        
        pbest = positions.copy()
        pbest_scores = np.array([self.fitness_function(p) for p in positions])
        gbest = pbest[np.argmax(pbest_scores)]
        
        for _ in range(self.iterations):
            for i in range(self.num_agents):
                r1, r2 = np.random.rand(), np.random.rand()
                velocities[i] = (w * velocities[i] + 
                                 c1 * r1 * (pbest[i] - positions[i]) + 
                                 c2 * r2 * (gbest - positions[i]))
                positions[i] = np.clip(positions[i] + velocities[i], 0, 1) 
                
                current_score = self.fitness_function(positions[i])
                if current_score > pbest_scores[i]:
                    pbest[i] = positions[i]
                    pbest_scores[i] = current_score
            gbest = pbest[np.argmax(pbest_scores)]

        final_adfi, final_sbeh, final_sperf = self.calculate_adfi(gbest, self.interval_data)
        
        print("\n=== OPTIMIZED ADFI MATHEMATICAL MODEL ===")
        print(f"Alpha Weights (D_i): {np.round(gbest[0:5], 3)}")
        print(f"Beta Weights (I_j):  {np.round(gbest[5:10], 3)}")
        print(f"Fusion Weights (w1, w2): {np.round(gbest[10:12], 3)}")
        print(f"Calculated S_beh: {final_sbeh:.3f}")
        print(f"Calculated S_perf: {final_sperf:.3f}")
        print(f"Final ADFI Score: {final_adfi:.3f}")
        return gbest

# ==========================================
# LAYER 1: MULTIMODAL PERCEPTION & CPT
# ==========================================
class MCPS_PerceptionLayer:
    def __init__(self, pose_model='pose_landmarker.task', face_model='face_landmarker.task'):
        print("[Layer 1] Initializing Edge-Computing Pipeline with Patient-Guided CPT...")
        
        self.window_size = 30
        self.spatial_history = deque(maxlen=self.window_size)
        self.time_history = deque(maxlen=self.window_size)
        
        # Interval Management
        self.interval_duration = 30 
        self.all_intervals_data = []
        self.reset_interval_metrics()
        
        # Patient-Guided State Machine
        self.system_state = "INSTRUCT_WELCOME" 
        self.cpt_phase = "WAITING"
        self.stimulus_start_time = 0
        self.phase_start_time = 0 
        self.next_stimulus_time = 0
        
        pose_base = python.BaseOptions(model_asset_path=pose_model)
        self.pose_options = vision.PoseLandmarkerOptions(
            base_options=pose_base, running_mode=vision.RunningMode.VIDEO)
            
        face_base = python.BaseOptions(model_asset_path=face_model)
        self.face_options = vision.FaceLandmarkerOptions(
            base_options=face_base, running_mode=vision.RunningMode.VIDEO)

    def reset_interval_metrics(self):
        self.current_interval_metrics = {
            "v_list": [], "a_list": [], "sv_list": [],
            "blinks": 0, "gaze_deviations": 0,
            "omissions": 0, "commissions": 0, 
            "reaction_times": [], "timeouts": 0
        }

    def process_kinematics(self, current_coords, timestamp, torso_length):
        self.spatial_history.append(current_coords)
        self.time_history.append(timestamp)
        
        if len(self.spatial_history) < 3: return
            
        hist_array = np.array(self.spatial_history)
        variance = float(np.var(hist_array, axis=0).sum() / (torso_length + 1e-6))
        
        dx = hist_array[-1] - hist_array[-2]
        dt = (self.time_history[-1] - self.time_history[-2]) + 1e-6
        velocity = float(np.linalg.norm(dx / dt) / (torso_length + 1e-6))
        
        dx_prev = hist_array[-2] - hist_array[-3]
        dt_prev = (self.time_history[-2] - self.time_history[-3]) + 1e-6
        v_prev = float(np.linalg.norm(dx_prev / dt_prev) / (torso_length + 1e-6))
        acceleration = float(abs(velocity - v_prev) / dt)
        
        self.current_interval_metrics["v_list"].append(velocity)
        self.current_interval_metrics["a_list"].append(acceleration)
        self.current_interval_metrics["sv_list"].append(variance)

    def process_oculomotor(self, face_landmarks, width, height):
        p_top = np.array([face_landmarks[159].x, face_landmarks[159].y])
        p_bot = np.array([face_landmarks[145].x, face_landmarks[145].y])
        p_in = np.array([face_landmarks[33].x, face_landmarks[33].y])
        p_out = np.array([face_landmarks[133].x, face_landmarks[133].y])
        
        eye_height = np.linalg.norm(p_top - p_bot)
        eye_width = np.linalg.norm(p_in - p_out)
        ear = eye_height / (eye_width + 1e-6)
        
        if ear < 0.2: 
            self.current_interval_metrics["blinks"] += 1

        pupil = np.array([face_landmarks[468].x, face_landmarks[468].y])
        eye_center = (p_in + p_out) / 2.0
        gaze_dist = np.linalg.norm(pupil - eye_center)
        
        if gaze_dist > (eye_width * 0.15):
            self.current_interval_metrics["gaze_deviations"] += 1

    def save_interval_data(self):
        v = self.current_interval_metrics["v_list"]
        a = self.current_interval_metrics["a_list"]
        sv = self.current_interval_metrics["sv_list"]
        rts = self.current_interval_metrics["reaction_times"]
        
        interval_data = {
            "velocity": float(np.mean(v)) if v else 0.0,
            "acceleration": float(np.mean(a)) if a else 0.0,
            "variance": float(np.mean(sv)) if sv else 0.0,
            "blinks": self.current_interval_metrics["blinks"] / 30.0, 
            "gaze_loss": self.current_interval_metrics["gaze_deviations"] / 900.0,
            "omissions": self.current_interval_metrics["omissions"],
            "commissions": self.current_interval_metrics["commissions"],
            "avg_rt": float(np.mean(rts)) if rts else 1.0,
            "var_rt": float(np.var(rts)) if len(rts) > 1 else 0.0,
            "timeouts": self.current_interval_metrics["timeouts"]
        }
        self.all_intervals_data.append(interval_data)
        self.reset_interval_metrics()

    def draw_centered_text(self, display, text_lines, y_start=200):
        for i, line in enumerate(text_lines):
            cv2.putText(display, line, (20, y_start + (i * 40)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def execute_cpt_logic(self, display, current_time, key, speed, prob):
        if current_time >= self.next_stimulus_time and self.cpt_phase == "WAITING":
            is_target = random.random() < prob
            self.cpt_phase = "TARGET" if is_target else "DISTRACTOR"
            self.stimulus_start_time = current_time
            self.next_stimulus_time = current_time + random.uniform(speed, speed + 1.5)
            
        if self.cpt_phase != "WAITING":
            c_color = (0, 255, 0) if self.cpt_phase == "TARGET" else ((0, 0, 255) if random.random() < 0.5 else (255, 0, 0))
            cv2.circle(display, (320, 240), 100, c_color, -1)
            
            if current_time - self.stimulus_start_time > (speed * 0.8):
                if self.cpt_phase == "TARGET":
                    self.current_interval_metrics["omissions"] += 1
                    self.current_interval_metrics["timeouts"] += 1
                self.cpt_phase = "WAITING"
                
        if key == ord(' '):
            if self.cpt_phase == "TARGET":
                self.current_interval_metrics["reaction_times"].append(current_time - self.stimulus_start_time)
                self.cpt_phase = "WAITING"
            elif self.cpt_phase == "DISTRACTOR":
                self.current_interval_metrics["commissions"] += 1
                self.cpt_phase = "WAITING"
            elif self.cpt_phase == "WAITING":
                self.current_interval_metrics["commissions"] += 1 

    def run_assessment(self, phase_duration=30):
        target_nodes = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
        
        with vision.PoseLandmarker.create_from_options(self.pose_options) as pose_landmarker, \
             vision.FaceLandmarker.create_from_options(self.face_options) as face_landmarker:
            
            # To use mobile camera, change 0 to the IP stream (e.g., 'http://192.168.1.100:8080/video')
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            print("[System] Medical Assessment Ready. Waiting for Patient to begin.")
            
            while cap.isOpened():
                current_time = time.perf_counter()
                
                ret, frame = cap.read()
                if not ret: break
                
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                
                display = np.zeros((480, 1280, 3), dtype=np.uint8)
                display[:, 640:] = frame
                cv2.line(display, (640, 0), (640, 480), (255, 255, 255), 2)
                
                # --- BACKGROUND KINEMATIC TRACKING ---
                pose_result = pose_landmarker.detect_for_video(mp_image, int(current_time * 1000))
                if pose_result.pose_landmarks:
                    p = pose_result.pose_landmarks[0]
                    mid_shoulder = np.array([(p[11].x+p[12].x)/2, (p[11].y+p[12].y)/2])
                    mid_hip = np.array([(p[23].x+p[24].x)/2, (p[23].y+p[24].y)/2])
                    torso_length = np.linalg.norm(mid_shoulder - mid_hip)
                    
                    coords = np.array([[p[i].x, p[i].y, p[i].z] for i in target_nodes])
                    self.process_kinematics(coords, current_time, torso_length)
                    
                    for idx in target_nodes:
                        cx, cy = int(p[idx].x * 640) + 640, int(p[idx].y * 480)
                        cv2.circle(display, (cx, cy), 5, (0, 255, 255), -1)

                face_result = face_landmarker.detect_for_video(mp_image, int(current_time * 1000))
                if face_result.face_landmarks:
                    self.process_oculomotor(face_result.face_landmarks[0], 640, 480)

                # --- PATIENT-GUIDED UI STATE MACHINE ---
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'): break
                
                if self.system_state == "INSTRUCT_WELCOME":
                    instructions = [
                        "WELCOME TO THE ATTENTION TEST",
                        "",
                        "RULES:",
                        "1. If you see a GREEN circle, press SPACEBAR.",
                        "2. If you see RED or BLUE, DO NOTHING.",
                        "",
                        "Press SPACEBAR to continue."
                    ]
                    self.draw_centered_text(display, instructions, 150)
                    if key == ord(' '): self.system_state = "INSTRUCT_PHASE1"

                elif self.system_state == "INSTRUCT_PHASE1":
                    instructions = [
                        "PHASE 1: BASELINE WARM-UP",
                        "",
                        "The test will now begin.",
                        "The circles will appear slowly.",
                        "",
                        "Press SPACEBAR when you are ready."
                    ]
                    self.draw_centered_text(display, instructions, 150)
                    if key == ord(' '): 
                        self.system_state = "RUN_PHASE1"
                        self.phase_start_time = current_time
                        self.next_stimulus_time = current_time + 2.0

                elif self.system_state == "RUN_PHASE1":
                    elapsed = current_time - self.phase_start_time
                    cv2.putText(display, f"PHASE 1 (Time Left: {phase_duration - int(elapsed)}s)", 
                                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
                    
                    self.execute_cpt_logic(display, current_time, key, speed=3.0, prob=1.0)
                    
                    if elapsed > phase_duration:
                        self.save_interval_data() 
                        self.system_state = "INSTRUCT_PHASE2"

                elif self.system_state == "INSTRUCT_PHASE2":
                    instructions = [
                        "PHASE 2: SUSTAINED ATTENTION",
                        "",
                        "Good job. The test is getting longer.",
                        "Keep your eyes on the screen.",
                        "",
                        "Press SPACEBAR to continue."
                    ]
                    self.draw_centered_text(display, instructions, 150)
                    if key == ord(' '): 
                        self.system_state = "RUN_PHASE2"
                        self.phase_start_time = current_time
                        self.next_stimulus_time = current_time + 2.0

                elif self.system_state == "RUN_PHASE2":
                    elapsed = current_time - self.phase_start_time
                    cv2.putText(display, f"PHASE 2 (Time Left: {phase_duration - int(elapsed)}s)", 
                                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
                    
                    self.execute_cpt_logic(display, current_time, key, speed=2.0, prob=1.0)
                    
                    if elapsed > phase_duration:
                        self.save_interval_data() 
                        self.system_state = "INSTRUCT_PHASE3"

                elif self.system_state == "INSTRUCT_PHASE3":
                    instructions = [
                        "PHASE 3: INHIBITORY CONTROL",
                        "",
                        "WARNING: Red and Blue distractors will now appear.",
                        "The speed will increase.",
                        "DO NOT press the spacebar for wrong colors.",
                        "",
                        "Press SPACEBAR to start final phase."
                    ]
                    self.draw_centered_text(display, instructions, 150)
                    if key == ord(' '): 
                        self.system_state = "RUN_PHASE3"
                        self.phase_start_time = current_time
                        self.next_stimulus_time = current_time + 2.0

                elif self.system_state == "RUN_PHASE3":
                    elapsed = current_time - self.phase_start_time
                    cv2.putText(display, f"PHASE 3 (Time Left: {phase_duration - int(elapsed)}s)", 
                                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
                    
                    self.execute_cpt_logic(display, current_time, key, speed=1.2, prob=0.4)
                    
                    if elapsed > phase_duration:
                        self.save_interval_data() 
                        break 

                cv2.imshow('MCPS Multimodal Assessment', display)
                        
            cap.release()
            cv2.destroyAllWindows()
            
        print("[Layer 1] Assessment Complete. Proceeding to Optimization.")
        return self.all_intervals_data


# ==========================================
# SYSTEM EXECUTION
# ==========================================
if __name__ == "__main__":
    print("=== STARTING MEDICAL CYBER-PHYSICAL SYSTEM ===")
    
    # 1. Run Layer 1 (Multi-Phase Guided CPT)
    perception = MCPS_PerceptionLayer()
    
    # Defaults to 3 phases of 30 seconds each
    intervals_history = perception.run_assessment(phase_duration=30)
    
    # 2. Layer 2 Mathematical Coefficient Optimization
    if intervals_history:
        optimizer = ADFI_Optimizer(interval_data=intervals_history, num_agents=20, iterations=30)
        final_adfi_weights = optimizer.run_pso()