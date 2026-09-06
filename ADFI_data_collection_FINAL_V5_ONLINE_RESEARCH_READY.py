import os
import time
import json
import random
import threading
import urllib.request
import io
import re
import zipfile
import hashlib
from collections import deque
from datetime import datetime, timezone

import av
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

import firebase_admin
from firebase_admin import credentials, firestore

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ============================================================
# RESEARCH CONFIGURATION & MATHEMATICAL THRESHOLDS
# ============================================================
TOTAL_TARGET_TRIALS = 480  
EXCLUDE_TRAINING_FROM_MATH = True  
SCORED_TRIALS = 440 if EXCLUDE_TRAINING_FROM_MATH else 480

THRESH_VELOCITY = 0.50          
THRESH_ACCEL = 0.50             
THRESH_GAZE_DEV_RATIO = 0.30    
THRESH_BLINK_COUNT = 3          
MIN_VALID_FRAME_RATIO = 0.50    

THRESH_TARDY_RT = 1.00          
INCLUDE_COMMISSION_IN_S_PERF = False  

MAX_INVALID_INTERVAL_RATIO = 0.20 

# ============================================================
# STREAMLIT CONFIG & MODELS
# ============================================================
st.set_page_config(page_title="ADFI Behavioral & Cognitive Research", layout="wide", initial_sidebar_state="collapsed")

POSE_MODEL = "pose_landmarker_lite.task"
FACE_MODEL = "face_landmarker.task"
POSE_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
FACE_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"

@st.cache_resource
def download_models():
    for path, url in [(POSE_MODEL, POSE_URL), (FACE_MODEL, FACE_URL)]:
        if not os.path.exists(path): urllib.request.urlretrieve(url, path)
download_models()

@st.cache_resource
def init_firestore():
    """
    Firebase configuration:
    1) Local development: firebase_credentials.json beside this script.
    2) Streamlit Cloud: [firebase] secret containing the service-account fields.
       Never commit firebase_credentials.json to GitHub.
    """
    try:
        # Streamlit Cloud / hosted deployment
        if "firebase" in st.secrets:
            raw = st.secrets["firebase"]
            service_account = dict(raw)
            if service_account.get("private_key"):
                service_account["private_key"] = service_account["private_key"].replace("\\n", "\n")
            cred = credentials.Certificate(service_account)
            if not firebase_admin._apps:
                firebase_admin.initialize_app(cred)
            return firestore.client(), None

        # Local development fallback
        if not os.path.exists("firebase_credentials.json"):
            return None, (
                "Firebase is not configured. For local testing place "
                "firebase_credentials.json beside the app; for Streamlit Cloud "
                "configure the [firebase] secret."
            )

        if not firebase_admin._apps:
            cred = credentials.Certificate("firebase_credentials.json")
            firebase_admin.initialize_app(cred)

        return firestore.client(), None

    except Exception as exc:
        return None, f"Firebase initialization failed: {type(exc).__name__}: {exc}"

db, firebase_error = init_firestore()

# ============================================================
# CPT PROTOCOL DEFINITION
# ============================================================
PHASES = [
    {"name": "TRAINING", "trials": 40, "stimulus_interval": 2.2, "target_probability": 1.00, "rule": "circle"},
    {"name": "FOCUS", "trials": 160, "stimulus_interval": 1.75, "target_probability": 0.50, "rule": "shape"},
    {"name": "CONTROL", "trials": 160, "stimulus_interval": 1.35, "target_probability": 0.50, "rule": "go_nogo"},
    {"name": "DISTRACTION", "trials": 120, "stimulus_interval": 1.05, "target_probability": 0.45, "rule": "distraction"},
]

def generate_cpt_trials():
    trials = []
    tid = 1
    for phase in PHASES:
        for _ in range(phase["trials"]):
            target = random.random() < phase["target_probability"]
            rule = phase["rule"]
            if rule == "circle": stim = "green_circle"
            elif rule == "shape": stim = "green_circle" if target else random.choice(["square", "triangle", "diamond"])
            elif rule == "go_nogo": stim = "green_circle" if target else "red_circle"
            else: stim = "green_circle" if target else random.choice(["square", "triangle", "diamond", "cross"])

            trials.append({
                "trial_id": tid,
                "phase": phase["name"],
                "target": target,
                "stimulus": stim,
                "interval": phase["stimulus_interval"] + random.uniform(0.1, 0.4),
                "scored": not (EXCLUDE_TRAINING_FROM_MATH and phase["name"] == "TRAINING")
            })
            tid += 1
    return trials

# ============================================================
# CAMERA / BIOMARKER PROCESSOR
# ============================================================
class CameraBiomarkerProcessor:
    def __init__(self):
        self.lock = threading.RLock()
        pose_options = vision.PoseLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=POSE_MODEL), running_mode=vision.RunningMode.VIDEO)
        face_options = vision.FaceLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=FACE_MODEL), running_mode=vision.RunningMode.VIDEO)
        self.pose_detector = vision.PoseLandmarker.create_from_options(pose_options)
        self.face_detector = vision.FaceLandmarker.create_from_options(face_options)
        
        self.running = False
        self.raw_observations = []
        self.spatial_history = deque(maxlen=3)
        self.time_history = deque(maxlen=3)
        self.last_blink_state = False
        self.blink_start_time = None

    def start_assessment(self):
        with self.lock:
            self.running = True
            self.raw_observations.clear()
            self.spatial_history.clear()
            self.time_history.clear()
            self.last_blink_state = False
            self.blink_start_time = None

    def stop_assessment(self):
        with self.lock: self.running = False

    def _process_kinematics(self, p, now):
        required = [0, 11, 12, 23, 24]
        coords = np.array([[p[i].x, p[i].y, p[i].z] for i in required])
        shoulder = np.array([(p[11].x + p[12].x) / 2, (p[11].y + p[12].y) / 2])
        hip = np.array([(p[23].x + p[24].x) / 2, (p[23].y + p[24].y) / 2])
        torso_length = np.linalg.norm(shoulder - hip) + 1e-6
        
        self.spatial_history.append(coords)
        self.time_history.append(now)
        if len(self.spatial_history) < 3: return 0.0, 0.0, 0.0

        arr = np.asarray(self.spatial_history)
        movement_variance = float(np.var(arr, axis=0).sum() / torso_length)
        dt = max(0.001, self.time_history[-1] - self.time_history[-2])
        prev_dt = max(0.001, self.time_history[-2] - self.time_history[-3])
        
        velocity = float(np.linalg.norm((arr[-1] - arr[-2]) / dt) / torso_length)
        prev_velocity = float(np.linalg.norm((arr[-2] - arr[-3]) / prev_dt) / torso_length)
        acceleration = float(abs(velocity - prev_velocity) / dt)
        return velocity, acceleration, movement_variance

    def _process_face(self, landmarks, current_time):
        top = np.array([landmarks[159].x, landmarks[159].y])
        bottom = np.array([landmarks[145].x, landmarks[145].y])
        inner = np.array([landmarks[133].x, landmarks[133].y])
        outer = np.array([landmarks[33].x, landmarks[33].y])
        ear = np.linalg.norm(top - bottom) / (np.linalg.norm(inner - outer) + 1e-6)
        
        currently_blinking = ear < 0.20
        prolonged_occlusion = False
        
        if currently_blinking:
            if self.blink_start_time is None:
                self.blink_start_time = current_time
            elif current_time - self.blink_start_time > 1.0: 
                prolonged_occlusion = True
        else:
            self.blink_start_time = None

        blink_trigger = currently_blinking and not self.last_blink_state
        self.last_blink_state = currently_blinking

        r_inner = np.array([landmarks[133].x, landmarks[133].y])
        r_outer = np.array([landmarks[33].x, landmarks[33].y])
        r_iris = np.array([landmarks[468].x, landmarks[468].y])
        r_gaze_ratio = np.linalg.norm(r_iris - r_outer) / (np.linalg.norm(r_inner - r_outer) + 1e-6)
        gaze_deviation = r_gaze_ratio < 0.35 or r_gaze_ratio > 0.65

        return blink_trigger, gaze_deviation, prolonged_occlusion

    def recv(self, frame: av.VideoFrame):
        try:
            img = frame.to_ndarray(format="bgr24")
            img = cv2.resize(img, (640, 480))
            now = time.time()  

            with self.lock: running = self.running
            if not running: return av.VideoFrame.from_ndarray(img, format="bgr24")

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(now * 1000)
            
            v, a, mv = 0.0, 0.0, 0.0
            blink, gaze_dev = False, False
            valid_pose, valid_face = False, False
            hands_covering_face = False

            try:
                pose_result = self.pose_detector.detect_for_video(mp_image, timestamp_ms)
                face_result = self.face_detector.detect_for_video(mp_image, timestamp_ms)

                if pose_result and pose_result.pose_landmarks:
                    valid_pose = True
                    p = pose_result.pose_landmarks[0]
                    v, a, mv = self._process_kinematics(p, now)
                    
                    nose = p[0]
                    for idx in [15, 16, 19, 20]: 
                        pt = p[idx]
                        if getattr(pt, 'visibility', 0) > 0.4:
                            dist = ((pt.x - nose.x)**2 + (pt.y - nose.y)**2)**0.5
                            if dist < 0.12: 
                                hands_covering_face = True
                                break

                if face_result and face_result.face_landmarks:
                    valid_face = True
                    blink, gaze_dev, prolonged_occlusion = self._process_face(face_result.face_landmarks[0], now)
                    if prolonged_occlusion:
                        hands_covering_face = True
            except Exception:
                pass 

            if hands_covering_face:
                valid_face = False

            with self.lock:
                self.raw_observations.append({
                    "timestamp": now,
                    "velocity": v,
                    "acceleration": a,
                    "movement_variance": mv,
                    "gaze_deviation": gaze_dev,
                    "blink": blink,
                    "valid_pose": valid_pose,
                    "valid_face": valid_face,
                    "valid_combined": valid_pose and valid_face
                })

            if not valid_pose or not valid_face:
                cv2.putText(img, "MEASUREMENT INVALID", (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 4)
            
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        
        except Exception:
            dummy = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(dummy, "RECOVERING FEED", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return av.VideoFrame.from_ndarray(dummy, format="bgr24")

# ============================================================
# DATA PROCESSING & SAVING LOGIC
# ============================================================
def calculate_I_j(trial_dict):
    if trial_dict.get("outcome") == "omission": return 1
    if trial_dict.get("reaction_time") is not None and trial_dict.get("reaction_time") > THRESH_TARDY_RT: return 1
    if INCLUDE_COMMISSION_IN_S_PERF and trial_dict.get("outcome") == "commission": return 1
    return 0

def calculate_D_i(interval_frames):
    if not interval_frames: return 0, False
    valid_frames = [f for f in interval_frames if f["valid_combined"]]
    if len(valid_frames) / len(interval_frames) < MIN_VALID_FRAME_RATIO:
        return 0, False  

    max_v = max(f["velocity"] for f in valid_frames)
    max_a = max(f["acceleration"] for f in valid_frames)
    blinks = sum(1 for f in valid_frames if f["blink"])
    gaze_dev_ratio = sum(1 for f in valid_frames if f["gaze_deviation"]) / len(valid_frames)

    if (max_v > THRESH_VELOCITY) or (max_a > THRESH_ACCEL) or (blinks >= THRESH_BLINK_COUNT) or (gaze_dev_ratio > THRESH_GAZE_DEV_RATIO):
        return 1, True
    return 0, True

def validate_cpt_dataset(trials):
    if not isinstance(trials, list):
        return False, "CPT results are not a list."

    if len(trials) != TOTAL_TARGET_TRIALS:
        return False, f"Expected {TOTAL_TARGET_TRIALS} trials, got {len(trials)}."

    ids = [t.get("trial_id") for t in trials]
    if any(not isinstance(x, int) for x in ids):
        return False, "One or more trial IDs are missing or not integers."

    if len(set(ids)) != len(ids):
        return False, "Duplicate trial IDs found."

    if sorted(ids) != list(range(1, TOTAL_TARGET_TRIALS + 1)):
        return False, f"Trial IDs are not sequential from 1 to {TOTAL_TARGET_TRIALS}."

    allowed_outcomes = {"hit", "omission", "commission", "correct_rejection"}

    for t in trials:
        if "phase" not in t or "target" not in t or "scored" not in t:
            return False, f"Trial {t.get('trial_id')} is missing required protocol fields."

        if t.get("outcome") not in allowed_outcomes:
            return False, (
                f"Trial {t.get('trial_id')} has invalid outcome "
                f"{t.get('outcome')!r}."
            )

        if t.get("scored") and t.get("outcome") == "pending":
            return False, f"Trial {t.get('trial_id')} is scored but still pending."

        if t.get("scored") and t.get("outcome") == "hit":
            if t.get("reaction_time") is None:
                return False, f"Trial {t.get('trial_id')} is a hit but has no reaction time."

    return True, "Valid"

def process_research_dataset(participant, session_id, trials, camera_frames):
    is_valid, error_msg = validate_cpt_dataset(trials)
    if not is_valid: raise ValueError(f"Strict Validation Failed: {error_msg}")
    
    processed_trials = []
    processed_intervals = []
    
    S_perf = 0
    S_beh = 0

    for i, t in enumerate(trials):
        tdict = dict(t) 
        
        if tdict["scored"]:
            tdict["I_j"] = calculate_I_j(tdict)
            S_perf += tdict["I_j"]
        else:
            tdict["I_j"] = None 
            
        processed_trials.append(tdict)

        start_ts = tdict.get("stimulus_onset_epoch", 0)
        if i + 1 < len(trials):
            end_ts = trials[i+1].get("stimulus_onset_epoch", start_ts + tdict["interval"])
        else:
            end_ts = start_ts + tdict["interval"]
        
        interval_frames = [f for f in camera_frames if start_ts <= f["timestamp"] < end_ts]
        D_i, is_interval_valid = calculate_D_i(interval_frames)
        
        if tdict["scored"] and is_interval_valid: 
            S_beh += D_i
        
        processed_intervals.append({
            "interval_id": tdict["trial_id"],
            "phase": tdict["phase"],
            "scored": tdict["scored"],
            "start_ts": start_ts,
            "end_ts": end_ts,
            "interval_valid": is_interval_valid,
            "D_i": D_i if tdict["scored"] else None,
            "mean_velocity": float(np.mean([f["velocity"] for f in interval_frames])) if interval_frames else 0,
            "mean_acceleration": float(np.mean([f["acceleration"] for f in interval_frames])) if interval_frames else 0,
            "movement_variance": float(np.mean([f["movement_variance"] for f in interval_frames])) if interval_frames else 0,
            "total_blinks": sum(1 for f in interval_frames if f["blink"]),
            "gaze_deviations": sum(1 for f in interval_frames if f["gaze_deviation"])
        })

    scored_trials = [t for t in processed_trials if t["scored"]]
    scored_intervals = [iv for iv in processed_intervals if iv["scored"]]

    hits = sum(1 for t in scored_trials if t.get("outcome") == "hit")
    omis = sum(1 for t in scored_trials if t.get("outcome") == "omission")
    coms = sum(1 for t in scored_trials if t.get("outcome") == "commission")
    rts = [t.get("reaction_time") for t in scored_trials if t.get("reaction_time") is not None]
    
    invalid_interval_count = sum(1 for iv in scored_intervals if not iv["interval_valid"])
    session_valid = (invalid_interval_count / SCORED_TRIALS) <= MAX_INVALID_INTERVAL_RATIO

    summary = {
        "session_valid": session_valid,
        "total_trials": TOTAL_TARGET_TRIALS,
        "scored_trials": SCORED_TRIALS,
        "hits": hits, "omissions": omis, "commissions": coms,
        "mean_rt_ms": float(np.mean(rts) * 1000) if rts else 0,
        "rt_sd_ms": float(np.std(rts) * 1000) if rts else 0,
        "mean_velocity": float(np.mean([iv["mean_velocity"] for iv in scored_intervals])),
        "mean_acceleration": float(np.mean([iv["mean_acceleration"] for iv in scored_intervals])),
        "movement_variance": float(np.mean([iv["movement_variance"] for iv in scored_intervals])),
        "gaze_deviations": sum(iv["gaze_deviations"] for iv in scored_intervals),
        "total_blinks": sum(iv["total_blinks"] for iv in scored_intervals),
        "invalid_intervals": invalid_interval_count,
        "total_frames": len(camera_frames),
        "valid_pose_frames": sum(1 for f in camera_frames if f["valid_pose"]),
        "valid_face_frames": sum(1 for f in camera_frames if f["valid_face"]),
        "valid_combined_frames": sum(1 for f in camera_frames if f["valid_combined"]),
        "S_beh": S_beh,
        "S_perf": S_perf
    }
    return summary, processed_trials, processed_intervals


def firestore_safe(value):
    """Recursively convert NumPy/Pandas values to Firestore-safe Python values."""
    if isinstance(value, dict):
        return {str(k): firestore_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [firestore_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [firestore_safe(v) for v in value.tolist()]
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        v = float(value)
        return v if np.isfinite(v) else None
    if value is None:
        return None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    try:
        result = pd.isna(value)
        if isinstance(result, (bool, np.bool_)) and bool(result):
            return None
    except (TypeError, ValueError):
        pass
    return value


def chunked_batch_write(db, refs_and_data):
    batch = db.batch()
    count = 0
    for ref, data in refs_and_data:
        batch.set(ref, firestore_safe(data))
        count += 1
        if count >= 450:
            batch.commit()
            batch = db.batch()
            count = 0
    if count > 0: batch.commit()

def save_to_firebase(participant, session_id, summary, trials, intervals, raw_frames):
    """
    Save one complete research session.

    Collections:
      participants/{participant_id}
      sessions/{session_id}
        raw_trials/{trial_XXX}
        behavioral_intervals/{interval_XXX}
        raw_camera_observations/{frame_XXXXXX}
    """
    if db is None:
        return False, firebase_error or "Firebase is not connected."

    try:
        participant_id = participant["id"].strip()

        participant_doc = firestore_safe({
            "participant_id": participant_id,
            "age": int(participant["age"]),
            "gender": participant["gender"],
            "group": participant["group"],
            "last_tested": firestore.SERVER_TIMESTAMP
        })
        db.collection("participants").document(participant_id).set(
            participant_doc, merge=True
        )

        session_ref = db.collection("sessions").document(session_id)

        session_doc = firestore_safe({
            "participant_id": participant_id,
            "session_id": session_id,
            "age": int(participant["age"]),
            "gender": participant["gender"],
            "group": participant["group"],
            "participant": {
                "participant_id": participant_id,
                "age": int(participant["age"]),
                "gender": participant["gender"],
                "group": participant["group"],
            },
            "timestamp": firestore.SERVER_TIMESTAMP,
            "status": "completed",
            "session_valid": bool(summary["session_valid"]),
            "summary": summary,
            "protocol": {
                "total_trials": TOTAL_TARGET_TRIALS,
                "training_trials": 40,
                "scored_trials": SCORED_TRIALS,
                "exclude_training_from_math": EXCLUDE_TRAINING_FROM_MATH,
                "max_invalid_interval_ratio": MAX_INVALID_INTERVAL_RATIO
            }
        })
        session_ref.set(session_doc)

        writes = []

        for t in trials:
            writes.append((
                session_ref.collection("raw_trials").document(
                    f"trial_{int(t['trial_id']):03d}"
                ),
                t
            ))

        for iv in intervals:
            writes.append((
                session_ref.collection("behavioral_intervals").document(
                    f"interval_{int(iv['interval_id']):03d}"
                ),
                iv
            ))

        for idx, frame in enumerate(raw_frames):
            writes.append((
                session_ref.collection("raw_camera_observations").document(
                    f"frame_{idx:06d}"
                ),
                frame
            ))

        chunked_batch_write(db, writes)

        # Verify the parent session exists after all writes.
        verification = session_ref.get()
        if not verification.exists:
            return False, "Firebase write returned but session verification failed."

        return True, (
            f"Firebase saved successfully: {len(trials)} trials, "
            f"{len(intervals)} intervals, {len(raw_frames)} camera observations."
        )

    except Exception as exc:
        return False, f"Firebase save failed: {type(exc).__name__}: {exc}"

def save_to_local_pc(participant, session_id, summary, trials, intervals, raw_frames):
    """Save the complete research session locally on the same PC running Streamlit.

    The save location is anchored to this Python file, not the current working
    directory, so the data will consistently be stored beside the application.

    A complete JSON file is written for every session and contains:
      - Participant ID
      - Age
      - Gender
      - Research group
      - Session ID
      - Complete calculated summary
      - All CPT trials
      - All behavioral intervals
      - All camera-derived observations

    CSV files are also retained for analysis.
    """
    try:
        app_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.join(app_dir, "local_dataset")
        os.makedirs(base_dir, exist_ok=True)

        participant_safe = firestore_safe({
            "participant_id": str(participant.get("id", "")),
            "age": int(participant.get("age", 0)),
            "gender": str(participant.get("gender", "")),
            "group": str(participant.get("group", "")),
        })

        summary_safe = firestore_safe(summary)
        trials_safe = firestore_safe(trials)
        intervals_safe = firestore_safe(intervals)
        raw_frames_safe = firestore_safe(raw_frames)

        # --------------------------------------------------------
        # 1. Master CSV summary
        # --------------------------------------------------------
        summary_path = os.path.join(base_dir, "master_sessions_summary.csv")
        summary_row = {
            "participant_id": participant_safe["participant_id"],
            "age": participant_safe["age"],
            "gender": participant_safe["gender"],
            "group": participant_safe["group"],
            "session_id": session_id,
            **summary_safe,
        }
        df_summary = pd.DataFrame([summary_row])
        if not os.path.exists(summary_path):
            df_summary.to_csv(summary_path, index=False)
        else:
            df_summary.to_csv(summary_path, mode="a", header=False, index=False)

        # --------------------------------------------------------
        # 2. Individual CSV files
        # --------------------------------------------------------
        pd.DataFrame(trials_safe).to_csv(
            os.path.join(base_dir, f"{session_id}_trials.csv"), index=False
        )
        pd.DataFrame(intervals_safe).to_csv(
            os.path.join(base_dir, f"{session_id}_intervals.csv"), index=False
        )
        pd.DataFrame(raw_frames_safe).to_csv(
            os.path.join(base_dir, f"{session_id}_camera_observations.csv"),
            index=False
        )

        # --------------------------------------------------------
        # 3. COMPLETE JSON DATASET -- main research backup
        # --------------------------------------------------------
        complete_dataset = {
            "schema_version": "ADFI-Research-v4",
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            "participant": participant_safe,
            "session": {
                "session_id": str(session_id),
                "participant_id": participant_safe["participant_id"],
                "age": participant_safe["age"],
                "gender": participant_safe["gender"],
                "group": participant_safe["group"],
            },
            "summary": summary_safe,
            "raw_trials": trials_safe,
            "behavioral_intervals": intervals_safe,
            "raw_camera_observations": raw_frames_safe,
        }

        json_path = os.path.join(
            base_dir, f"{session_id}_COMPLETE_DATA.json"
        )
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(complete_dataset, f, ensure_ascii=False, indent=2)

        # --------------------------------------------------------
        # 4. Participant metadata JSON (easy to inspect separately)
        # --------------------------------------------------------
        participant_json_path = os.path.join(
            base_dir, f"{participant_safe['participant_id']}_participant.json"
        )
        with open(participant_json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "participant_id": participant_safe["participant_id"],
                    "age": participant_safe["age"],
                    "gender": participant_safe["gender"],
                    "group": participant_safe["group"],
                    "latest_session_id": str(session_id),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        return True
    except Exception as e:
        print(f"Local save error: {type(e).__name__}: {e}")
        return False


# ============================================================
# PARTICIPANT DOWNLOAD PACKAGE
# ============================================================
def build_complete_download_package(participant, session_id, summary, trials, intervals, raw_frames):
    """
    Build one ZIP containing the participant's complete research record.

    The participant can download this ZIP to their own computer and send it
    to the researcher. It contains:
      - README / participant session receipt
      - complete JSON dataset
      - participant metadata JSON
      - trials CSV
      - behavioral intervals CSV
      - camera observations CSV
      - SHA-256 checksums
    """
    participant_safe = firestore_safe({
        "participant_id": str(participant.get("id", "")),
        "age": int(participant.get("age", 0)),
        "gender": str(participant.get("gender", "")),
        "group": str(participant.get("group", "")),
    })
    summary_safe = firestore_safe(summary)
    trials_safe = firestore_safe(trials)
    intervals_safe = firestore_safe(intervals)
    raw_frames_safe = firestore_safe(raw_frames)

    complete_dataset = {
        "schema_version": "ADFI-Research-v5",
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "participant": participant_safe,
        "session": {
            "session_id": str(session_id),
            "participant_id": participant_safe["participant_id"],
            "age": participant_safe["age"],
            "gender": participant_safe["gender"],
            "group": participant_safe["group"],
        },
        "summary": summary_safe,
        "protocol": {
            "total_trials": TOTAL_TARGET_TRIALS,
            "training_trials": 40,
            "scored_trials": SCORED_TRIALS,
            "exclude_training_from_math": EXCLUDE_TRAINING_FROM_MATH,
            "max_invalid_interval_ratio": MAX_INVALID_INTERVAL_RATIO,
            "include_commission_in_S_perf": INCLUDE_COMMISSION_IN_S_PERF,
            "threshold_tardy_rt_seconds": THRESH_TARDY_RT,
            "threshold_velocity": THRESH_VELOCITY,
            "threshold_acceleration": THRESH_ACCEL,
            "threshold_gaze_deviation_ratio": THRESH_GAZE_DEV_RATIO,
            "threshold_blink_count": THRESH_BLINK_COUNT,
            "min_valid_frame_ratio": MIN_VALID_FRAME_RATIO,
        },
        "raw_trials": trials_safe,
        "behavioral_intervals": intervals_safe,
        "raw_camera_observations": raw_frames_safe,
    }

    complete_json = json.dumps(
        complete_dataset, ensure_ascii=False, indent=2
    ).encode("utf-8")

    participant_json = json.dumps(
        {
            **participant_safe,
            "session_id": str(session_id),
        },
        ensure_ascii=False, indent=2
    ).encode("utf-8")

    trials_csv = pd.DataFrame(trials_safe).to_csv(index=False).encode("utf-8")
    intervals_csv = pd.DataFrame(intervals_safe).to_csv(index=False).encode("utf-8")
    camera_csv = pd.DataFrame(raw_frames_safe).to_csv(index=False).encode("utf-8")

    receipt = f"""ADFI RESEARCH ASSESSMENT — PARTICIPANT DATA RECEIPT
==========================================================

Participant ID : {participant_safe["participant_id"]}
Age            : {participant_safe["age"]}
Gender         : {participant_safe["gender"]}
Research Group : {participant_safe["group"]}
Session ID     : {session_id}

Assessment status : COMPLETED
Total CPT trials  : {TOTAL_TARGET_TRIALS}
Scored trials     : {SCORED_TRIALS}
S_beh             : {summary_safe.get("S_beh")}
S_perf            : {summary_safe.get("S_perf")}
Session valid     : {summary_safe.get("session_valid")}

FILES IN THIS PACKAGE
---------------------
1. COMPLETE_DATA.json
   Complete research record for this session.

2. PARTICIPANT_METADATA.json
   Participant ID, age, gender, group and session ID.

3. TRIALS.csv
   All {TOTAL_TARGET_TRIALS} CPT trials.

4. BEHAVIORAL_INTERVALS.csv
   Behavioral interval records aligned to the CPT protocol.

5. CAMERA_OBSERVATIONS.csv
   Camera-derived behavioral observations. This file contains
   derived observations/features, not the participant's video recording.

6. SHA256SUMS.txt
   Integrity checksums for all files.

IMPORTANT
---------
Keep this ZIP file private and send it only to the authorized researcher.
Do not post it publicly or share it on social media.
The application does not diagnose ADHD; the dataset is for research/assessment
and later statistical/ML validation.
"""
    receipt_bytes = receipt.encode("utf-8")

    files = {
        "README_SESSION_RECEIPT.txt": receipt_bytes,
        "COMPLETE_DATA.json": complete_json,
        "PARTICIPANT_METADATA.json": participant_json,
        "TRIALS.csv": trials_csv,
        "BEHAVIORAL_INTERVALS.csv": intervals_csv,
        "CAMERA_OBSERVATIONS.csv": camera_csv,
    }

    checksum_lines = []
    for name, data in files.items():
        checksum_lines.append(
            f"{hashlib.sha256(data).hexdigest()}  {name}"
        )
    checksums = ("\n".join(checksum_lines) + "\n").encode("utf-8")
    files["SHA256SUMS.txt"] = checksums

    buffer = io.BytesIO()
    with zipfile.ZipFile(
        buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as zf:
        for name, data in files.items():
            zf.writestr(name, data)

    return buffer.getvalue()


# ============================================================
# ADFI CPT CUSTOM STREAMLIT COMPONENT
# ============================================================
CPT_COMPONENT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    ".adfi_cpt_component"
)
CPT_COMPONENT_HTML = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
html,body{margin:0;padding:0;width:100%;height:100%;overflow:hidden;background:#fafafa;font-family:Arial,sans-serif}
#app{box-sizing:border-box;width:100%;height:470px;border:1px solid #e5e7eb;border-radius:14px;background:#fff;position:relative;outline:none;user-select:none}
#top{position:absolute;left:20px;right:20px;top:16px;display:flex;justify-content:space-between;font-size:14px;font-weight:600}
#phase{color:#2563eb} #counter{color:#6b7280}
#center{position:absolute;left:20px;right:20px;top:55px;bottom:60px;display:flex;align-items:center;justify-content:center}
#message{text-align:center;color:#374151;font-size:25px;font-weight:700;line-height:1.5}
#stimulus{width:140px;height:140px;display:none}
#footer{position:absolute;left:0;right:0;bottom:16px;text-align:center;color:#9ca3af;font-size:14px}
#status{position:absolute;left:20px;right:20px;bottom:42px;text-align:center;color:#6b7280;font-size:12px}
.green{background:#10b981;border-radius:50%}.red{background:#ef4444;border-radius:50%}
.square{background:#6b7280;border-radius:8px}.diamond{background:#6b7280;border-radius:8px;transform:rotate(45deg)}
.triangle{width:0!important;height:0!important;border-left:70px solid transparent;border-right:70px solid transparent;border-bottom:140px solid #6b7280}
.cross{width:140px;height:140px;display:flex;align-items:center;justify-content:center;color:#374151;font-size:125px;font-weight:700}
</style>
</head>
<body>
<div id="app" tabindex="0">
 <div id="top"><div id="phase">Loading…</div><div id="counter"></div></div>
 <div id="center"><div id="message">Loading assessment…</div><div id="stimulus"></div></div>
 <div id="footer">Click this box once, then use SPACEBAR to respond.</div>
 <div id="status"></div>
</div>
<script>
(function(){
  const app=document.getElementById("app"), phase=document.getElementById("phase"),
        counter=document.getElementById("counter"), msg=document.getElementById("message"),
        stim=document.getElementById("stimulus"), footer=document.getElementById("footer"),
        status=document.getElementById("status");
  let trials=[],results=[],idx=0,responded=false,running=false,finished=false,onsetPerf=0,stimTimer=null,intTimer=null;

  function value(v){window.parent.postMessage({isStreamlitMessage:true,type:"streamlit:setComponentValue",value:v},"*")}
  function height(){window.parent.postMessage({isStreamlitMessage:true,type:"streamlit:setFrameHeight",height:470},"*")}
  function clearTimers(){if(stimTimer)clearTimeout(stimTimer);if(intTimer)clearTimeout(intTimer);stimTimer=intTimer=null}
  function reset(){stim.className="";stim.style.display="none";stim.style.width="140px";stim.style.height="140px";stim.style.transform="none";stim.innerHTML=""}
  function draw(s){
    reset();stim.style.display="block";
    if(s==="circle"||s==="green_circle")stim.className="green";
    else if(s==="red_circle")stim.className="red";
    else if(s==="square")stim.className="square";
    else if(s==="diamond")stim.className="diamond";
    else if(s==="triangle")stim.className="triangle";
    else if(s==="cross"){stim.className="cross";stim.textContent="×"}
  }
  function phaseText(p){
    if(p==="TRAINING")phase.textContent="TRAINING — Press SPACE for the green circle";
    else if(p==="FOCUS")phase.textContent="FOCUS — Press SPACE for the green circle";
    else if(p==="CONTROL")phase.textContent="CONTROL — Press SPACE for GREEN, ignore RED";
    else phase.textContent="DISTRACTION — Press SPACE for the green circle";
  }
  function finishTrial(t){
    if(!t.outcome||t.outcome==="pending"){
      t.reaction_time=null;t.response_epoch=null;t.response_performance=null;
      t.outcome=t.target?"omission":"correct_rejection";
    }
    results.push(t);idx++;
    if(idx>=trials.length)finish();else next();
  }
  function next(){
    if(finished)return;
    clearTimers();
    const t=trials[idx];responded=false;
    t.outcome="pending";t.reaction_time=null;t.response_epoch=null;t.response_performance=null;
    counter.textContent="Trial "+(idx+1)+" / "+trials.length;phaseText(t.phase);
    msg.style.display="none";draw(t.stimulus);
    requestAnimationFrame(function(){
      if(finished)return;
      t.stimulus_onset_epoch=Date.now()/1000;
      t.stimulus_onset_performance=performance.now();
      onsetPerf=t.stimulus_onset_performance;
      stimTimer=setTimeout(function(){
        stim.style.display="none";
        if(!responded){
          t.reaction_time=null;t.response_epoch=null;t.response_performance=null;
          t.outcome=t.target?"omission":"correct_rejection";
        }
      },1500);
      intTimer=setTimeout(function(){finishTrial(t)},Math.max(1500,Number(t.interval)*1000));
    });
  }
  function key(e){
    if(e.code!=="Space"||!running||finished||responded)return;
    e.preventDefault();e.stopPropagation();
    const now=performance.now(),rt=(now-onsetPerf)/1000;
    if(rt<0.05||rt>1.50)return;
    const t=trials[idx];responded=true;
    t.response_epoch=Date.now()/1000;t.response_performance=now;t.reaction_time=rt;
    t.outcome=t.target?"hit":"commission";stim.style.display="none";
  }
  function finish(){
    if(finished)return;finished=true;running=false;clearTimers();
    document.removeEventListener("keydown",key,true);window.removeEventListener("keydown",key,true);
    reset();footer.style.display="none";msg.style.display="block";
    msg.innerHTML="Assessment Complete!<br><span style='font-size:16px;font-weight:normal'>"+
      results.length+" / "+trials.length+" trials completed.<br>Sending results to Streamlit…</span>";
    status.textContent="Please wait — do not refresh the page.";
    value({status:"completed",total_trials:results.length,results:results});
  }
  function start(){
    if(running||finished)return;
    if(!Array.isArray(trials)||trials.length!==480){msg.textContent="Protocol error: expected 480 trials.";return}
    running=true;idx=0;results=[];app.focus();
    let n=3;msg.textContent="Assessment starting in "+n+"…";
    const timer=setInterval(function(){n--;if(n>0)msg.textContent="Assessment starting in "+n+"…";else{clearInterval(timer);next()}},1000);
  }
  function render(e){
    if(!e.data||e.data.type!=="streamlit:render")return;
    const a=e.data.args||{};
    if(Array.isArray(a.trials))trials=a.trials;
    height();if(!running&&!finished)start();
  }
  document.addEventListener("keydown",key,true);
  window.addEventListener("keydown",key,true);
  app.addEventListener("mousedown",function(){app.focus()});
  app.addEventListener("click",function(){app.focus()});
  window.addEventListener("message",render);
  height();

  // Tell Streamlit that the component is ready only after every
  // event handler has been registered.
  window.parent.postMessage({
    isStreamlitMessage:true,
    type:"streamlit:componentReady",
    apiVersion:1
  },"*");
})();
</script>
</body>
</html>
"""

def ensure_cpt_component():
    os.makedirs(CPT_COMPONENT_DIR, exist_ok=True)
    path = os.path.join(CPT_COMPONENT_DIR, "index.html")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(CPT_COMPONENT_HTML)
    return path

ensure_cpt_component()

adfi_cpt_component = components.declare_component(
    "adfi_cpt_research_component",
    path=CPT_COMPONENT_DIR
)

# ============================================================
# STREAMLIT UI
# ============================================================
for key, default in {
    "session_id": None,
    "cpt_trials": generate_cpt_trials(),
    "camera_ready": False,
    "data_processed": False,
    "saved": False,
    "finished": False,
    "started": False,
    "download_package": None,
    "local_saved": False,
    "firebase_saved": False,
    "firebase_save_message": "",
    "save_attempted": False,
    "researcher_authenticated": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ---------- Premium, participant-friendly styling ----------
st.markdown("""
<style>
.block-container {max-width: 1200px; padding-top: 2rem; padding-bottom: 3rem;}
.adfi-card {
    border: 1px solid rgba(128,128,128,.25);
    border-radius: 18px;
    padding: 22px;
    margin: 8px 0 16px 0;
    background: rgba(128,128,128,.05);
}
.adfi-step {
    border-radius: 14px;
    padding: 12px 14px;
    border: 1px solid rgba(128,128,128,.22);
    min-height: 82px;
}
.adfi-small {font-size: .88rem; opacity: .78;}
.adfi-success {
    border-left: 5px solid #22c55e;
    padding: 14px 16px;
    border-radius: 10px;
    background: rgba(34,197,94,.08);
}
</style>
""", unsafe_allow_html=True)

st.title("ADFI Behavioral & Cognitive Research Assessment")
st.caption("Secure research data collection • 480 CPT trials • Webcam-derived behavioral observations")

st.markdown("""
<div class="adfi-card">
<b>Before you begin</b><br>
Please use a laptop or desktop if possible. Sit in a quiet, well-lit room,
keep your face and upper body visible, and keep the browser tab open until
the assessment is finished. Your browser will request camera permission.
<br><br>
<b>Privacy:</b> Use your assigned Participant ID. Do not enter your full name
unless your approved research protocol specifically requires it.
</div>
""", unsafe_allow_html=True)

# Researcher access is intentionally locked behind authentication.
tabs = st.tabs(["🧑‍💻 Participant Assessment", "🔐 Researcher Dashboard"])

rtc_config = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {
            "urls": [st.secrets["turn"]["url"]],
            "username": st.secrets["turn"]["username"],
            "credential": st.secrets["turn"]["credential"],
        }
    ]
})

# ============================================================
# PARTICIPANT ASSESSMENT
# ============================================================
with tabs[0]:

    if not st.session_state.started:

        st.markdown("## Step 1 — Participant Registration")

        cols = st.columns(3)
        with cols[0]:
            st.markdown('<div class="adfi-step"><b>1. Register</b><br><span class="adfi-small">Enter your assigned ID and demographics.</span></div>', unsafe_allow_html=True)
        with cols[1]:
            st.markdown('<div class="adfi-step"><b>2. Complete task</b><br><span class="adfi-small">Allow camera access and complete all 480 trials.</span></div>', unsafe_allow_html=True)
        with cols[2]:
            st.markdown('<div class="adfi-step"><b>3. Download</b><br><span class="adfi-small">Download the complete ZIP and send it to the researcher.</span></div>', unsafe_allow_html=True)

        st.markdown("### Participant information")

        with st.form("reg_form"):
            pid = st.text_input(
                "Participant ID *",
                placeholder="Example: P001",
                help="Use the ID supplied by the researcher. Do not use your full name."
            )

            c1, c2 = st.columns(2)
            with c1:
                age = st.number_input(
                    "Age *",
                    min_value=5,
                    max_value=100,
                    value=18,
                    step=1
                )
            with c2:
                gender = st.selectbox(
                    "Gender *",
                    ["Male", "Female", "Other", "Prefer not to say"]
                )

            group = st.selectbox(
                "Research Group *",
                ["Control", "Experimental/ADHD", "Unspecified"],
                help="Select only the group assigned by the researcher."
            )

            st.markdown("### Consent and setup")
            consent = st.checkbox(
                "I understand the research instructions and consent to collection "
                "and processing of my cognitive-task and webcam-derived behavioral "
                "data according to the study information provided by the researcher."
            )

            st.caption(
                "If the participant is below the applicable age of consent, "
                "the research team must obtain the required guardian/participant "
                "consent under its approved protocol before starting."
            )

            submitted = st.form_submit_button(
                "Continue to Camera & Assessment →",
                type="primary",
                use_container_width=True
            )

        if submitted:
            pid = pid.strip().replace("/", "_").replace("\\", "_").replace(" ", "_")

            if not pid:
                st.error("Please enter your assigned Participant ID.")
            elif len(pid) > 64:
                st.error("Participant ID must be 64 characters or fewer.")
            elif not consent:
                st.error("Please confirm consent before starting.")
            else:
                duplicate = False
                if db is not None:
                    try:
                        duplicate = (
                            db.collection("participants")
                            .document(pid)
                            .get()
                            .exists
                        )
                    except Exception:
                        # Do not block a participant solely because the
                        # duplicate check is temporarily unavailable.
                        duplicate = False

                if duplicate:
                    st.error(
                        f"Participant ID {pid} already exists. "
                        "Please contact the researcher if you need to repeat a session."
                    )
                else:
                    st.session_state.participant = {
                        "id": pid,
                        "age": int(age),
                        "gender": gender,
                        "group": group
                    }
                    st.session_state.session_id = (
                        f"{pid}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
                    )
                    st.session_state.cpt_trials = generate_cpt_trials()
                    st.session_state.camera_ready = False
                    st.session_state.data_processed = False
                    st.session_state.saved = False
                    st.session_state.finished = False
                    st.session_state.component_result = None
                    st.session_state.local_saved = False
                    st.session_state.firebase_saved = False
                    st.session_state.firebase_save_message = ""
                    st.session_state.save_attempted = False
                    st.session_state.download_package = None
                    st.session_state.started = True
                    st.session_state.assessment_component_key = (
                        f"cpt_{int(time.time()*1000000)}"
                    )
                    st.rerun()

    elif st.session_state.started and not st.session_state.finished:

        st.markdown("## Step 2 — Camera Check & Assessment")
        st.info(
            "Keep your face and upper body visible. Do not switch tabs, "
            "refresh the page, or close the browser during the assessment."
        )

        left, right = st.columns([2.1, 1.0], gap="large")

        with right:
            st.markdown("### 📷 Camera Biomarker Capture")

            ctx = webrtc_streamer(
                key="camera_" + st.session_state.session_id,
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=rtc_config,
                video_processor_factory=CameraBiomarkerProcessor,
                media_stream_constraints={
                    "video": {
                        "facingMode": "user",
                        "width": {"ideal": 640},
                        "height": {"ideal": 480},
                        "frameRate": {"ideal": 15}
                    },
                    "audio": False
                },
                async_processing=True
            )

            if ctx and ctx.video_processor:
                st.session_state.proc = ctx.video_processor
                if not ctx.video_processor.running:
                    ctx.video_processor.start_assessment()
                if len(ctx.video_processor.raw_observations) >= 10:
                    st.session_state.camera_ready = True

            if st.session_state.camera_ready:
                st.success("Camera measurement stream is active.")
            else:
                st.warning(
                    "Allow camera access and position yourself so the face "
                    "and upper body remain visible."
                )

            st.caption(
                "The research dataset stores derived camera observations/features, "
                "not a saved video recording."
            )

        with left:
            st.markdown("### 🧠 Continuous Performance Assessment")

            if not st.session_state.camera_ready:
                st.warning(
                    "Please allow the camera and wait until the camera stream "
                    "is initialized. The task can then proceed."
                )

            result = adfi_cpt_component(
                trials=st.session_state.cpt_trials,
                height=470,
                key=st.session_state.assessment_component_key
            )

            if isinstance(result, dict):
                status = result.get("status")
                received = result.get("results", [])

                if status == "completed":
                    valid, message = validate_cpt_dataset(received)

                    if valid:
                        st.session_state.component_result = result
                        st.session_state.completed_trials = received
                        st.session_state.finished = True
                        st.rerun()
                    else:
                        st.error("CPT validation failed: " + message)

            st.caption(
                "Keyboard: SPACEBAR • Reaction time captured in the browser "
                "with performance.now() • Total: 480 trials"
            )

    # ========================================================
    # COMPLETED ASSESSMENT / AUTOMATIC SAVE
    # ========================================================
    elif st.session_state.finished:

        if not st.session_state.data_processed:

            proc = st.session_state.get("proc")
            if proc is not None:
                proc.stop_assessment()
                with proc.lock:
                    raw_observations = list(proc.raw_observations)
            else:
                raw_observations = []

            try:
                summary, trials, intervals = process_research_dataset(
                    st.session_state.participant,
                    st.session_state.session_id,
                    st.session_state.completed_trials,
                    raw_observations
                )

                st.session_state.summary = summary
                st.session_state.trials_data = trials
                st.session_state.intervals_data = intervals
                st.session_state.raw_frames = raw_observations
                st.session_state.data_processed = True

            except Exception as exc:
                st.error(
                    f"Processing failed: {type(exc).__name__}: {exc}"
                )
                st.exception(exc)
                st.stop()

        # Automatic permanent/server-side save.
        # This runs once per completed session.
        if not st.session_state.save_attempted:
            st.session_state.save_attempted = True

            local_ok = save_to_local_pc(
                st.session_state.participant,
                st.session_state.session_id,
                st.session_state.summary,
                st.session_state.trials_data,
                st.session_state.intervals_data,
                st.session_state.raw_frames
            )
            st.session_state.local_saved = bool(local_ok)

            firebase_ok, firebase_message = save_to_firebase(
                st.session_state.participant,
                st.session_state.session_id,
                st.session_state.summary,
                st.session_state.trials_data,
                st.session_state.intervals_data,
                st.session_state.raw_frames
            )
            st.session_state.firebase_saved = bool(firebase_ok)
            st.session_state.firebase_save_message = firebase_message
            st.session_state.saved = bool(firebase_ok)

        summary = st.session_state.summary

        st.markdown("## Step 3 — Assessment Completed")
        st.success(
            "🎉 Your assessment has been completed. Your research record "
            "has been processed successfully."
        )

        a,b,c,d,e = st.columns(5)
        a.metric("Trials", summary["total_trials"])
        b.metric("Scored", summary["scored_trials"])
        c.metric("Hits", summary["hits"])
        d.metric("Omissions", summary["omissions"])
        e.metric("S_perf", summary["S_perf"])

        a,b,c,d = st.columns(4)
        a.metric("S_beh", summary["S_beh"])
        b.metric("Mean RT", f"{summary['mean_rt_ms']:.2f} ms")
        c.metric("Invalid Intervals", summary["invalid_intervals"])
        d.metric("Camera Observations", summary["total_frames"])

        if summary["session_valid"]:
            st.success("✅ Session passed the research quality gate.")
        else:
            st.error(
                "⚠️ Session failed the research quality gate. "
                "Please contact the researcher before repeating the session."
            )

        # Save status
        st.markdown("### ☁️ Research server status")
        if st.session_state.firebase_saved:
            st.success("Firebase: complete session uploaded successfully.")
        else:
            st.error(
                "Firebase upload did not complete. "
                "Your download package is still available below. "
                "Send the package to the researcher and ask them to retry the upload."
            )
            st.caption(st.session_state.firebase_save_message)

            if st.button("🔄 Retry Firebase Upload", use_container_width=True):
                ok, msg = save_to_firebase(
                    st.session_state.participant,
                    st.session_state.session_id,
                    st.session_state.summary,
                    st.session_state.trials_data,
                    st.session_state.intervals_data,
                    st.session_state.raw_frames
                )
                st.session_state.firebase_saved = bool(ok)
                st.session_state.saved = bool(ok)
                st.session_state.firebase_save_message = msg
                st.rerun()

        if st.session_state.local_saved:
            st.info(
                "Server-side local backup was also written where the Streamlit "
                "application is running. For online participants, use the ZIP "
                "download below as your personal local copy."
            )

        with st.expander("Complete calculated session summary"):
            st.json(summary)

        # Build one participant-owned download package.
        if st.session_state.download_package is None:
            try:
                st.session_state.download_package = build_complete_download_package(
                    st.session_state.participant,
                    st.session_state.session_id,
                    st.session_state.summary,
                    st.session_state.trials_data,
                    st.session_state.intervals_data,
                    st.session_state.raw_frames
                )
            except Exception as exc:
                st.session_state.download_package = None
                st.error(
                    f"Could not prepare the download package: "
                    f"{type(exc).__name__}: {exc}"
                )

        st.markdown("---")
        st.markdown("## 📦 Final step — Save your research record to your computer")

        st.markdown("""
<div class="adfi-card">
<b>Important:</b> Click the download button below and save the ZIP file.
This is your complete local copy. It contains the participant information,
session summary, all 480 CPT trials, behavioral intervals, camera-derived
observations, and an integrity checksum file.
<br><br>
After downloading, <b>send the ZIP file to the researcher using the method
they provided</b>. Do not upload it publicly.
</div>
""", unsafe_allow_html=True)

        if st.session_state.download_package is not None:
            safe_pid = re.sub(
                r"[^A-Za-z0-9_-]+", "_",
                str(st.session_state.participant["id"])
            )
            download_name = f"ADFI_{safe_pid}_{st.session_state.session_id.split('_',1)[-1]}_COMPLETE_DATA.zip"

            st.download_button(
                "⬇️ Download COMPLETE Research Data Package (ZIP)",
                data=st.session_state.download_package,
                file_name=download_name,
                mime="application/zip",
                type="primary",
                use_container_width=True,
                help="Save this ZIP to your computer. It contains every research record for this session."
            )

        st.caption(
            "Browser security requires the participant to click the download "
            "button; the website cannot silently write files into a user's PC."
        )

        st.markdown("### Session receipt")
        st.code(
            f"Participant ID: {st.session_state.participant['id']}\n"
            f"Session ID: {st.session_state.session_id}\n"
            f"Firebase saved: {st.session_state.firebase_saved}\n"
            f"Session valid: {summary['session_valid']}",
            language="text"
        )

        st.markdown("---")
        if st.button(
            "Start New Participant",
            use_container_width=True,
            key="new_participant"
        ):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

# ============================================================
# RESEARCHER DASHBOARD — PASSWORD PROTECTED
# ============================================================
with tabs[1]:

    st.header("Researcher Dashboard")
    st.caption("Restricted access — participant data is not shown without authentication.")

    # Password can be supplied through Streamlit secrets:
    # [researcher]
    # password = "CHANGE_THIS"
    # or environment variable ADFI_RESEARCHER_PASSWORD.
    configured_password = os.environ.get("ADFI_RESEARCHER_PASSWORD", "")
    try:
        if not configured_password and "researcher" in st.secrets:
            configured_password = str(st.secrets["researcher"]["password"])
    except Exception:
        pass

    if not st.session_state.researcher_authenticated:
        if not configured_password:
            st.error(
                "Researcher dashboard is locked because no researcher password "
                "has been configured."
            )
            st.info(
                "Configure ADFI_RESEARCHER_PASSWORD or the Streamlit "
                "[researcher] password secret before using this dashboard."
            )
        else:
            with st.form("researcher_login"):
                password = st.text_input(
                    "Researcher password",
                    type="password"
                )
                login = st.form_submit_button(
                    "🔐 Sign in",
                    type="primary",
                    use_container_width=True
                )

            if login:
                if password == configured_password:
                    st.session_state.researcher_authenticated = True
                    st.rerun()
                else:
                    st.error("Incorrect password.")

    else:
        top1, top2 = st.columns([4,1])
        with top1:
            st.success("Authenticated researcher session.")
        with top2:
            if st.button("Sign out"):
                st.session_state.researcher_authenticated = False
                st.rerun()

        if firebase_error:
            st.error("Firebase: " + firebase_error)

        if db:
            if st.button(
                "🔄 Load Dataset from Firebase",
                key="load_firebase_dataset"
            ):
                try:
                    sessions = []

                    for doc in db.collection("sessions").stream():
                        d = doc.to_dict()
                        sessions.append({
                            "participant_id": d.get("participant_id"),
                            "session_id": d.get("session_id"),
                            "group": d.get("group"),
                            "session_valid": d.get("session_valid", "Unknown"),
                            **d.get("summary", {})
                        })

                    if sessions:
                        df = pd.DataFrame(sessions)

                        st.dataframe(df, use_container_width=True)

                        if "session_valid" in df.columns:
                            valid_df = df[
                                df["session_valid"] == True
                            ]
                        else:
                            valid_df = df

                        st.success(
                            f"Found {len(valid_df)} valid sessions "
                            f"out of {len(df)} total."
                        )

                        st.download_button(
                            "⬇️ Download VALID Master Dataset CSV",
                            data=valid_df.to_csv(index=False),
                            file_name="ADFI_Master_Dataset_VALID.csv",
                            mime="text/csv"
                        )

                        buffer = io.BytesIO()
                        with pd.ExcelWriter(
                            buffer,
                            engine="openpyxl"
                        ) as writer:
                            df.to_excel(
                                writer,
                                index=False,
                                sheet_name="All_Sessions"
                            )
                            valid_df.to_excel(
                                writer,
                                index=False,
                                sheet_name="Valid_Sessions"
                            )

                        st.download_button(
                            "⬇️ Download Master Dataset Excel",
                            data=buffer.getvalue(),
                            file_name="ADFI_Master_Dataset.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    else:
                        st.info("No completed sessions found in Firebase.")

                except Exception as exc:
                    st.error(
                        f"Firebase dashboard error: "
                        f"{type(exc).__name__}: {exc}"
                    )
        else:
            st.warning(
                "Firebase is not connected. Participant local downloads remain available."
            )

