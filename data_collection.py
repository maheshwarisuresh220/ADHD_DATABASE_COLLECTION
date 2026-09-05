import os
import time
import json
import random
import threading
import urllib.request
import io
import re
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
    if not os.path.exists("firebase_credentials.json"):
        return None, "firebase_credentials.json was not found in the app folder."

    try:
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

def chunked_batch_write(db, refs_and_data):
    batch = db.batch()
    count = 0
    for ref, data in refs_and_data:
        batch.set(ref, data)
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

        db.collection("participants").document(participant_id).set({
            "participant_id": participant_id,
            "age": int(participant["age"]),
            "gender": participant["gender"],
            "group": participant["group"],
            "last_tested": firestore.SERVER_TIMESTAMP
        }, merge=True)

        session_ref = db.collection("sessions").document(session_id)

        session_ref.set({
            "participant_id": participant_id,
            "session_id": session_id,
            "group": participant["group"],
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
    try:
        base_dir = "local_dataset"
        os.makedirs(base_dir, exist_ok=True)
        
        summary_path = os.path.join(base_dir, "master_sessions_summary.csv")
        summary_row = {"participant_id": participant["id"], "session_id": session_id, "group": participant["group"], **summary}
        df_summary = pd.DataFrame([summary_row])
        if not os.path.exists(summary_path):
            df_summary.to_csv(summary_path, index=False)
        else:
            df_summary.to_csv(summary_path, mode='a', header=False, index=False)
            
        pd.DataFrame(trials).to_csv(
            os.path.join(base_dir, f"{session_id}_trials.csv"), index=False
        )
        pd.DataFrame(intervals).to_csv(
            os.path.join(base_dir, f"{session_id}_intervals.csv"), index=False
        )
        pd.DataFrame(raw_frames).to_csv(
            os.path.join(base_dir, f"{session_id}_camera_observations.csv"),
            index=False
        )
        return True
    except Exception as e:
        print(f"Local save error: {e}")
        return False

# ============================================================
# BROWSER -> STREAMLIT CPT BRIDGE
# ============================================================
def consume_cpt_bridge():
    """
    Read the browser JSON bridge only after Streamlit has received it.
    This is deliberately separate from rendering the textarea so the
    completion path is easy to debug.
    """
    raw = st.session_state.get("cpt_results", "")

    if not isinstance(raw, str) or not raw.strip():
        return False, "No CPT results received yet."

    try:
        trials = json.loads(raw)
    except json.JSONDecodeError as exc:
        return False, f"Browser sent invalid JSON: {exc}"

    valid, message = validate_cpt_dataset(trials)
    if not valid:
        return False, message

    st.session_state.completed_trials = trials
    st.session_state.finished = True
    st.session_state.bridge_received = True

    return True, f"{len(trials)} CPT trials received and validated."


# ============================================================
# STREAMLIT UI APP
# ============================================================
if "session_id" not in st.session_state: st.session_state.session_id = None
if "cpt_trials" not in st.session_state: st.session_state.cpt_trials = generate_cpt_trials()
if "camera_ready" not in st.session_state: st.session_state.camera_ready = False
if "auto_pid" not in st.session_state: st.session_state.auto_pid = f"P_{int(time.time())}"
if "data_processed" not in st.session_state: st.session_state.data_processed = False
if "bridge_received" not in st.session_state: st.session_state.bridge_received = False
if "saved" not in st.session_state: st.session_state.saved = False

# Hard CSS Block to lock camera interactibility and visually hide the data text_area
if st.session_state.get("started") and not st.session_state.get("finished"):
    st.markdown("""
    <style>
    div[data-testid="stTextArea"] {
        position: absolute !important;
        left: -9999px !important;
    }
    [data-testid='stWebRtc'] { opacity: 0.95; }
    </style>
    """, unsafe_allow_html=True)

st.title("ADFI Behavioral and Cognitive Research Assessment")
st.caption(
    "Research data collection mode: 480 CPT trials = 40 training + 440 scored. "
    "Firefly/PSO/ANN/XAI are intentionally not executed during collection."
)
tabs = st.tabs(["Data Collection", "Researcher Dashboard"])

rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

with tabs[0]:
    if not st.session_state.get("started"):
        st.header("Research Participant Registration")
        with st.form("reg_form"):
            pid = st.text_input("Participant ID (Auto-generated or type your own)", value=st.session_state.auto_pid)
            age = st.number_input("Age", min_value=1, max_value=100)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            group = st.selectbox("Research Group", ["Control", "Experimental/ADHD", "Unspecified"])
            
            st.markdown("### Participant Consent (Data Handling)")
            consent_text = "I consent to the collection and processing of my kinematic and cognitive data for research purposes. I understand that this data will be anonymized, stored securely on a local PC and cloud database, and handled in compliance with privacy protocols."
            consent = st.checkbox(consent_text)
            
            if st.form_submit_button("Start Session"):
                if not consent: 
                    st.error("Consent must be verified before proceeding.")
                elif not pid.strip():
                    st.error("Participant ID is required. Please enter a valid ID.")
                else:
                    if db:
                        doc = db.collection("participants").document(pid).get()
                        if doc.exists:
                            st.error(f"Participant ID {pid} already exists. Use a unique ID.")
                            st.stop()

                    st.session_state.participant = {"id": pid.strip(), "age": age, "gender": gender, "group": group}
                    st.session_state.session_id = f"{pid.strip()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    st.session_state.camera_ready = False
                    st.session_state.started = True
                    st.rerun()

    elif st.session_state.started and not st.session_state.get("finished"):

        # ----------------------------------------------------------
        # IMPORTANT:
        # Register the Streamlit widget before reading its value.
        # The browser JS writes into this exact textarea.
        # ----------------------------------------------------------
        st.text_area(
            "Hidden Data Bridge",
            key="cpt_results",
            label_visibility="collapsed",
            height=1
        )

        # If the previous browser run already delivered the JSON,
        # consume it and move to the processing screen.
        raw_res = st.session_state.get("cpt_results", "")
        if isinstance(raw_res, str) and raw_res.strip() and not st.session_state.get("bridge_received"):
            ok, message = consume_cpt_bridge()
            if ok:
                st.rerun()

        col1, col2 = st.columns([2, 1])

        with col2:
            st.markdown("**Webcam Sync**")

            ctx = webrtc_streamer(
                key="cam",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=rtc_config,
                video_processor_factory=CameraBiomarkerProcessor,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True
            )

            if ctx and ctx.video_processor:
                st.session_state.proc = ctx.video_processor

                if not ctx.video_processor.running:
                    ctx.video_processor.start_assessment()

        with col1:
            proc = st.session_state.get("proc")

            if not st.session_state.camera_ready:
                st.warning(
                    "Waiting for camera initialization... "
                    "Please ensure your face and upper body are visible."
                )

                if proc and len(proc.raw_observations) > 10:
                    st.session_state.camera_ready = True
                    st.rerun()
                else:
                    time.sleep(0.5)
                    st.rerun()

            else:

                # --------------------------------------------------
                # The processing button now has a REAL Python action.
                # It does not depend on a JS auto-click.
                # --------------------------------------------------
                st.markdown("<br>", unsafe_allow_html=True)

                if st.button(
                    "Step 2: Complete Assessment & Process Data",
                    key="cpt_submit",
                    type="primary",
                    use_container_width=True
                ):
                    raw_res = st.session_state.get("cpt_results", "")

                    if not raw_res or not raw_res.strip():
                        st.error(
                            "❌ No CPT results have reached Streamlit yet. "
                            "Wait until Trial 480 shows 'Assessment Complete', "
                            "then click this button once."
                        )
                    else:
                        try:
                            completed_trials = json.loads(raw_res)
                            valid, message = validate_cpt_dataset(completed_trials)

                            if not valid:
                                st.error(f"❌ CPT validation failed: {message}")
                            else:
                                st.session_state.completed_trials = completed_trials
                                st.session_state.finished = True
                                st.session_state.bridge_received = True
                                st.rerun()

                        except json.JSONDecodeError as exc:
                            st.error(f"❌ Invalid browser data: {exc}")

                # --------------------------------------------------
                # Browser-native CPT engine
                # --------------------------------------------------
                trials_json = json.dumps(
                    st.session_state.cpt_trials,
                    separators=(",", ":")
                )
                render_id = re.sub(
                    r"[^A-Za-z0-9_-]",
                    "_",
                    str(st.session_state.session_id)
                )

                js_code = f"""
                <div id="cpt-display-{render_id}" style="
                    position:relative;
                    height:380px;
                    display:flex;
                    flex-direction:column;
                    justify-content:center;
                    align-items:center;
                    background:#fafafa;
                    border:1px solid #eaeaea;
                    border-radius:12px;
                    margin-bottom:20px;
                    user-select:none;
                ">
                    <div id="trial-counter" style="
                        position:absolute;
                        top:15px;
                        right:20px;
                        font-weight:bold;
                        color:#888;
                    "></div>

                    <div id="phase-indicator" style="
                        position:absolute;
                        top:15px;
                        left:20px;
                        font-weight:bold;
                        color:#3b82f6;
                    "></div>

                    <div id="cpt-msg" style="
                        font-size:24px;
                        font-weight:bold;
                        color:#555;
                        text-align:center;
                    "></div>

                    <div id="stimulus-element" style="
                        display:none;
                        width:140px;
                        height:140px;
                    "></div>

                    <div id="instruction-footer" style="
                        position:absolute;
                        bottom:15px;
                        font-weight:500;
                        color:#aaa;
                    ">
                        Press SPACEBAR to respond
                    </div>
                </div>

                <script>
                (function() {{
                    const root = document.getElementById("cpt-display-{render_id}");
                    if (!root || root.dataset.engineStarted === "1") return;
                    root.dataset.engineStarted = "1";

                    const trials = {trials_json};
                    const results = [];

                    let currentTrial = 0;
                    let hasResponded = false;
                    let testFinished = false;
                    let onsetPerformance = 0;

                    let stimTimeout = null;
                    let intervalTimeout = null;

                    const el = root.querySelector("#stimulus-element");
                    const msgEl = root.querySelector("#cpt-msg");
                    const counterEl = root.querySelector("#trial-counter");
                    const phaseEl = root.querySelector("#phase-indicator");
                    const footEl = root.querySelector("#instruction-footer");

                    function clearTimers() {{
                        if (stimTimeout !== null) clearTimeout(stimTimeout);
                        if (intervalTimeout !== null) clearTimeout(intervalTimeout);
                        stimTimeout = null;
                        intervalTimeout = null;
                    }}

                    function renderStimulus(stim) {{
                        el.style.display = "block";
                        el.style.width = "140px";
                        el.style.height = "140px";
                        el.style.borderRadius = "0";
                        el.style.transform = "none";
                        el.style.border = "none";
                        el.style.borderLeft = "none";
                        el.style.borderRight = "none";
                        el.style.borderBottom = "none";
                        el.innerHTML = "";
                        el.style.backgroundColor = "transparent";

                        if (stim === "circle" || stim === "green_circle") {{
                            el.style.backgroundColor = "#10b981";
                            el.style.borderRadius = "50%";
                        }} else if (stim === "red_circle") {{
                            el.style.backgroundColor = "#ef4444";
                            el.style.borderRadius = "50%";
                        }} else if (stim === "square") {{
                            el.style.backgroundColor = "#6b7280";
                        }} else if (stim === "triangle") {{
                            el.style.width = "0";
                            el.style.height = "0";
                            el.style.borderLeft = "70px solid transparent";
                            el.style.borderRight = "70px solid transparent";
                            el.style.borderBottom = "140px solid #6b7280";
                        }} else if (stim === "diamond") {{
                            el.style.backgroundColor = "#6b7280";
                            el.style.transform = "rotate(45deg)";
                        }} else if (stim === "cross") {{
                            el.innerHTML =
                                '<div style="font-size:120px;color:#374151;text-align:center;line-height:140px;font-weight:bold;">×</div>';
                        }}
                    }}

                    function finishTrial(t) {{
                        if (t.outcome === undefined || t.outcome === "pending") {{
                            t.response_epoch = null;
                            t.response_performance = null;
                            t.reaction_time = null;
                            t.outcome = t.target
                                ? "omission"
                                : "correct_rejection";
                        }}

                        results.push(t);
                        currentTrial += 1;

                        if (currentTrial >= trials.length) {{
                            finishTest();
                        }} else {{
                            runNextTrial();
                        }}
                    }}

                    function runNextTrial() {{
                        if (testFinished) return;

                        if (currentTrial >= trials.length) {{
                            finishTest();
                            return;
                        }}

                        clearTimers();

                        const t = trials[currentTrial];
                        hasResponded = false;

                        t.outcome = "pending";
                        t.reaction_time = null;
                        t.response_epoch = null;
                        t.response_performance = null;

                        counterEl.innerText =
                            "Trial: " + (currentTrial + 1) + " / " + trials.length;

                        if (t.phase === "TRAINING") {{
                            phaseEl.innerText =
                                "TRAINING: Press Space for the Green Circle.";
                        }} else if (t.phase === "FOCUS") {{
                            phaseEl.innerText =
                                "FOCUS: Press Space for the Green Circle. Ignore others.";
                        }} else if (t.phase === "CONTROL") {{
                            phaseEl.innerText =
                                "CONTROL: Press Space for GREEN. Ignore Red.";
                        }} else if (t.phase === "DISTRACTION") {{
                            phaseEl.innerText =
                                "DISTRACTION: Press Space for the Green Circle.";
                        }}

                        renderStimulus(t.stimulus);

                        // Browser-native timestamps.
                        // performance.now() is used for reaction time.
                        // Date.now() gives Unix epoch for camera synchronization.
                        requestAnimationFrame(() => {{
                            if (testFinished) return;

                            t.stimulus_onset_epoch = Date.now() / 1000.0;
                            t.stimulus_onset_performance = performance.now();
                            onsetPerformance = t.stimulus_onset_performance;

                            stimTimeout = setTimeout(() => {{
                                el.style.display = "none";

                                if (!hasResponded) {{
                                    t.response_epoch = null;
                                    t.response_performance = null;
                                    t.reaction_time = null;
                                    t.outcome = t.target
                                        ? "omission"
                                        : "correct_rejection";
                                }}
                            }}, 1500);

                            intervalTimeout = setTimeout(() => {{
                                finishTrial(t);
                            }}, Math.max(1500, t.interval * 1000));
                        }});
                    }}

                    function keyHandler(e) {{
                        if (
                            e.code !== "Space" ||
                            testFinished ||
                            hasResponded ||
                            currentTrial >= trials.length
                        ) return;

                        e.preventDefault();

                        const nowPerf = performance.now();
                        const rt =
                            (nowPerf - onsetPerformance) / 1000.0;

                        if (rt < 0.05 || rt > 1.50) return;

                        const t = trials[currentTrial];

                        hasResponded = true;

                        t.response_epoch = Date.now() / 1000.0;
                        t.response_performance = nowPerf;
                        t.reaction_time = rt;
                        t.outcome = t.target ? "hit" : "commission";

                        el.style.display = "none";
                    }}

                    document.addEventListener("keydown", keyHandler, true);

                    function finishTest() {{
                        if (testFinished) return;

                        testFinished = true;
                        clearTimers();

                        document.removeEventListener(
                            "keydown",
                            keyHandler,
                            true
                        );

                        el.style.display = "none";
                        counterEl.style.display = "none";
                        phaseEl.style.display = "none";
                        footEl.style.display = "none";

                        msgEl.innerHTML =
                            "Assessment Complete!<br><br>" +
                            '<span style="font-size:16px;font-weight:normal;">' +
                            "480 trials completed. " +
                            "Your results are being transferred to the research system." +
                            "</span>";

                        // Exact Streamlit textarea selector.
                        // Do NOT select the last textarea on the page.
                        const doc = window.parent.document;

                        const targetArea =
                            doc.querySelector(
                                'textarea[aria-label="Hidden Data Bridge"]'
                            );

                        if (!targetArea) {{
                            msgEl.innerHTML =
                                "Assessment Complete.<br><br>" +
                                '<span style="color:#dc2626;font-size:16px;">' +
                                "Data bridge not found. Please use the " +
                                "processing button below." +
                                "</span>";
                            return;
                        }}

                        const payload = JSON.stringify(results);

                        try {{
                            const setter =
                                Object.getOwnPropertyDescriptor(
                                    window.HTMLTextAreaElement.prototype,
                                    "value"
                                ).set;

                            setter.call(targetArea, payload);

                            // Trigger React/Streamlit input handling.
                            targetArea.dispatchEvent(
                                new Event("input", {{bubbles:true}})
                            );
                            targetArea.dispatchEvent(
                                new Event("change", {{bubbles:true}})
                            );

                            msgEl.innerHTML =
                                "Assessment Complete!<br><br>" +
                                '<span style="font-size:16px;font-weight:normal;">' +
                                "480/480 trials completed.<br>" +
                                "Please click <b>Step 2: Complete Assessment & Process Data</b>." +
                                "</span>";

                        }} catch (err) {{
                            msgEl.innerHTML =
                                "Assessment Complete.<br><br>" +
                                '<span style="color:#dc2626;font-size:16px;">' +
                                "Transfer error: " + String(err) +
                                "</span>";
                        }}
                    }}

                    // Prevent the browser page from losing focus during
                    // the actual CPT sequence.
                    root.addEventListener("mousedown", () => {{
                        try {{ root.focus(); }} catch (_) {{}}
                    }});

                    let countdown = 3;

                    msgEl.innerText =
                        "Test starting in " + countdown + "...";

                    const cInterval = setInterval(() => {{
                        countdown -= 1;

                        if (countdown > 0) {{
                            msgEl.innerText =
                                "Test starting in " + countdown + "...";
                        }} else {{
                            clearInterval(cInterval);
                            msgEl.style.display = "none";
                            runNextTrial();
                        }}
                    }}, 1000);

                }})();
                </script>
                """

                components.html(js_code, height=450)

                # Small researcher-facing bridge diagnostic.
                bridge_len = len(
                    st.session_state.get("cpt_results", "") or ""
                )

                if bridge_len:
                    st.caption(
                        f"Browser → Streamlit data bridge detected "
                        f"({bridge_len:,} characters)."
                    )
                else:
                    st.caption(
                        "Waiting for browser CPT results. "
                        "This value should become non-zero after Trial 480."
                    )

    elif st.session_state.get("finished"):

        if not st.session_state.data_processed:

            proc = st.session_state.get("proc")

            if proc:
                proc.stop_assessment()
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

            except ValueError as ve:
                st.error(f"❌ {ve}")
                st.stop()

            except Exception as e:
                st.error(
                    f"❌ Processing Error: {type(e).__name__}: {e}"
                )
                st.exception(e)
                st.stop()

        if st.session_state.data_processed:

            summary = st.session_state.summary

            st.markdown("## Session Evaluation Complete")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Trials", summary["total_trials"])
            c2.metric("Scored Trials", summary["scored_trials"])
            c3.metric("S_beh", summary["S_beh"])
            c4.metric("S_perf", summary["S_perf"])

            if not summary["session_valid"]:
                st.error(
                    "❌ SESSION INVALID: Missing camera coverage exceeded "
                    "the maximum threshold. Do not use this session as a "
                    "valid mathematical-validation sample."
                )
            else:
                st.success("✅ Session Passed Quality Gate.")

            st.write("### Participant Summary")
            st.json(summary)

            st.markdown("---")

            if not st.session_state.get("saved"):

                st.info(
                    "The data has been processed successfully. "
                    "Click the button below to save the completed session "
                    "to both the local dataset and Firebase."
                )

                if st.button(
                    "🚀 Submit Assessment Data to Firestore",
                    type="primary",
                    use_container_width=True,
                    key="final_save_button"
                ):

                    with st.spinner(
                        "Saving research data locally and syncing to Firebase..."
                    ):

                        loc_success = save_to_local_pc(
                            st.session_state.participant,
                            st.session_state.session_id,
                            st.session_state.summary,
                            st.session_state.trials_data,
                            st.session_state.intervals_data,
                            st.session_state.raw_frames
                        )

                        fb_success, fb_message = save_to_firebase(
                            st.session_state.participant,
                            st.session_state.session_id,
                            st.session_state.summary,
                            st.session_state.trials_data,
                            st.session_state.intervals_data,
                            st.session_state.raw_frames
                        )

                        if loc_success:
                            st.success(
                                "📁 Local dataset saved successfully."
                            )
                        else:
                            st.error(
                                "❌ Local dataset save failed. "
                                "Check the terminal for details."
                            )

                        if fb_success:
                            st.success(f"☁️ {fb_message}")
                            st.session_state.saved = True
                        else:
                            st.error(f"☁️ {fb_message}")
                            st.session_state.saved = False

                        if loc_success and fb_success:
                            st.success(
                                "✅ COMPLETE: This participant's session "
                                "is securely stored."
                            )

            else:
                st.success("✓ Assessment Data Saved Successfully.")

                st.write("### Saved Session")
                st.write(
                    f"Participant: `{st.session_state.participant['id']}`"
                )
                st.write(
                    f"Session: `{st.session_state.session_id}`"
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


with tabs[1]:
    st.header("Master Data Export")

    if firebase_error:
        st.error(f"Firebase status: {firebase_error}")

    if db:
        if st.button("Load Dataset from Firebase"):
            sessions = []
            docs = db.collection("sessions").stream()
            for doc in docs:
                d = doc.to_dict()
                row = {
                    "participant_id": d["participant_id"], "session_id": d["session_id"], "group": d["group"],
                    "session_valid": d.get("session_valid", "Unknown"), **d.get("summary", {})
                }
                sessions.append(row)
                
            if sessions:
                df = pd.DataFrame(sessions)
                st.dataframe(df)
                
                valid_df = df[df["session_valid"] == True] if "session_valid" in df.columns else df
                st.success(f"Found {len(valid_df)} Valid Sessions out of {len(df)} total.")

                csv = valid_df.to_csv(index=False)
                st.download_button("Download VALID Master Dataset (CSV)", data=csv, file_name="ADFI_Master_Dataset.csv", mime="text/csv")
                
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    df.to_excel(writer, index=False, sheet_name='All_Sessions')
                    valid_df.to_excel(writer, index=False, sheet_name='Valid_Sessions_Only')
                st.download_button("Download Master Dataset (Excel)", data=buffer.getvalue(), file_name="ADFI_Master_Dataset.xlsx", mime="application/vnd.ms-excel")
            else: st.info("No completed sessions found.")
    else: st.warning("Firebase not connected.")