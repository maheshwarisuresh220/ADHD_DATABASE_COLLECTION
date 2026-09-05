import os
import time
import json
import random
import socket
import threading
import urllib.request
from collections import deque
from datetime import datetime, timezone

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

import firebase_admin
from firebase_admin import credentials, firestore

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# FOCUS MISSION — MULTIMODAL ADHD-RELATED ATTENTION ASSESSMENT
# ============================================================
# This is a research/screening prototype, NOT a stand-alone
# diagnostic instrument for ADHD.
#
# DESIGN:
#   Patient information
#          ↓
#   Camera / biomarker recording  ───────────────┐
#          ↓                                    │
#   Separate Focus Mission stimulus panel       │
#          ↓                                    │
#   Trial performance                            │
#          ↓                                    │
#   Behavioral + oculomotor biomarkers ─────────┤
#          ↓                                    │
#   Firestore participant document <────────────┘
#
# The participant ID is used as the Firestore document ID
# (primary key). Biomarkers are stored with the participant
# demographics and assessment session.
#
# For large video files, use Firebase Storage rather than
# Firestore. This prototype stores the local recording path
# in the database and saves the camera recording locally.


# ============================================================
# STREAMLIT CONFIG
# ============================================================
st.set_page_config(
    page_title="Focus Mission",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# MODEL DOWNLOAD
# ============================================================
POSE_MODEL = "pose_landmarker_lite.task"
FACE_MODEL = "face_landmarker.task"

POSE_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)

# MediaPipe Face Landmarker model.
FACE_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)


def download_if_missing(path, url, label):
    if not os.path.exists(path):
        with st.spinner(f"Preparing {label}..."):
            urllib.request.urlretrieve(url, path)


download_if_missing(
    POSE_MODEL,
    POSE_URL,
    "movement tracking model",
)

download_if_missing(
    FACE_MODEL,
    FACE_URL,
    "eye tracking model",
)


# ============================================================
# FIREBASE
# ============================================================
@st.cache_resource
def init_firestore():
    """
    Expects firebase_credentials.json in the project directory.

    Firestore structure:
        participants/{participant_id}
            demographics
            latest_assessment
            assessments/{session_id}
                biomarkers
                performance
                trials
                recording
    """
    if not os.path.exists("firebase_credentials.json"):
        return None

    try:
        if not firebase_admin._apps:
            cred = credentials.Certificate(
                "firebase_credentials.json"
            )
            firebase_admin.initialize_app(cred)

        return firestore.client()

    except Exception as exc:
        st.warning(f"Firebase initialization failed: {exc}")
        return None


db = init_firestore()


# ============================================================
# GENERAL HELPERS
# ============================================================
def get_local_ip():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    try:
        sock.connect(("10.255.255.255", 1))
        return sock.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        sock.close()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def mean(values):
    return float(np.mean(values)) if values else None


def std(values):
    return float(np.std(values)) if values else None


def ratio(a, b):
    return float(a / b) if b else 0.0


def sanitize_id(value):
    """
    Firestore document IDs cannot contain '/'.
    Keep participant IDs simple and readable.
    """
    return value.strip().replace("/", "_")


# ============================================================
# ASSESSMENT DESIGN
# ============================================================
# Approximately 7 minutes.
# Training is deliberately easier.
# Later phases progressively increase cognitive demand.

PHASES = [
    {
        "id": 1,
        "name": "TRAINING",
        "title": "Learn the Mission",
        "duration": 60,
        "stimulus_interval": 2.6,
        "target_probability": 1.00,
        "rule": "circle",
        "description": "Respond to the green circle.",
    },
    {
        "id": 2,
        "name": "FOCUS",
        "title": "Stay Focused",
        "duration": 120,
        "stimulus_interval": 1.75,
        "target_probability": 0.50,
        "rule": "shape",
        "description": "Respond only to the circle. Ignore other shapes.",
    },
    {
        "id": 3,
        "name": "CONTROL",
        "title": "Control Your Response",
        "duration": 120,
        "stimulus_interval": 1.35,
        "target_probability": 0.50,
        "rule": "go_nogo",
        "description": "Respond to green. Do not respond to red.",
    },
    {
        "id": 4,
        "name": "DISTRACTION",
        "title": "Focus Under Distraction",
        "duration": 120,
        "stimulus_interval": 1.05,
        "target_probability": 0.45,
        "rule": "distraction",
        "description": "Find the circle while visual distractions appear.",
    },
]

TOTAL_DURATION = sum(p["duration"] for p in PHASES)


# ============================================================
# SESSION STATE
# ============================================================
defaults = {
    "participant_id": "",
    "name": "",
    "gender": "",
    "age": None,
    "session_id": "",
    "assessment_started": False,
    "assessment_finished": False,
    "saved_to_firebase": False,
    "last_results": None,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# TRIAL MODEL
# ============================================================
class Trial:
    def __init__(
        self,
        phase,
        target,
        stimulus,
        onset,
    ):
        self.phase = phase
        self.target = target
        self.stimulus = stimulus
        self.onset = onset
        self.response = None
        self.reaction_time = None
        self.outcome = "pending"

    def to_dict(self):
        return {
            "phase": self.phase,
            "target": self.target,
            "stimulus": self.stimulus,
            "onset": self.onset,
            "response": self.response,
            "reaction_time": self.reaction_time,
            "outcome": self.outcome,
        }


# ============================================================
# CAMERA / BIOMARKER PROCESSOR
# ============================================================
class CameraBiomarkerProcessor:
    """
    Camera window only.

    It does NOT draw the cognitive test over the camera.
    The participant sees the camera separately from the
    Focus Mission stimulus panel.

    Measurements:
      - pose velocity
      - pose acceleration
      - movement variance
      - blink count
      - gaze deviation count
      - valid/invalid pose frames

    The processor also records the incoming camera frames to
    a local MP4 file during the assessment.
    """

    def __init__(self):
        self.lock = threading.RLock()

        pose_base = python.BaseOptions(
            model_asset_path=POSE_MODEL
        )

        pose_options = vision.PoseLandmarkerOptions(
            base_options=pose_base,
            running_mode=vision.RunningMode.VIDEO,
            min_pose_detection_confidence=0.60,
            min_pose_presence_confidence=0.60,
        )

        face_base = python.BaseOptions(
            model_asset_path=FACE_MODEL
        )

        face_options = vision.FaceLandmarkerOptions(
            base_options=face_base,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.50,
            min_face_presence_confidence=0.50,
        )

        self.pose_detector = (
            vision.PoseLandmarker.create_from_options(
                pose_options
            )
        )

        self.face_detector = (
            vision.FaceLandmarker.create_from_options(
                face_options
            )
        )

        self.running = False
        self.start_time = None
        self.last_timestamp_ms = None

        self.recording_path = None
        self.writer = None

        self.spatial_history = deque(maxlen=20)
        self.time_history = deque(maxlen=20)

        self.last_blink_state = False
        self.last_gaze_deviation = False

        self.metrics = self._new_metrics()

    @staticmethod
    def _new_metrics():
        return {
            "velocity": [],
            "acceleration": [],
            "movement_variance": [],
            "blink_count": 0,
            "gaze_deviation_count": 0,
            "valid_pose_frames": 0,
            "invalid_pose_frames": 0,
            "face_frames": 0,
        }

    def start_assessment(self, recording_path):
        with self.lock:
            self.running = True
            self.start_time = time.perf_counter()
            self.last_timestamp_ms = None

            self.spatial_history.clear()
            self.time_history.clear()

            self.last_blink_state = False
            self.last_gaze_deviation = False

            self.metrics = self._new_metrics()

            self.recording_path = recording_path

            # Recording settings can be changed according to deployment.
            self.writer = cv2.VideoWriter(
                self.recording_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                20.0,
                (640, 480),
            )

    def stop_assessment(self):
        with self.lock:
            self.running = False

            if self.writer is not None:
                self.writer.release()
                self.writer = None

    def _process_pose(self, landmarks, now):
        required = [0, 11, 12, 23, 24]

        visible = True

        for idx in required:
            point = landmarks[idx]

            if hasattr(point, "visibility"):
                if point.visibility < 0.55:
                    visible = False
                    break

        if not visible:
            self.metrics["invalid_pose_frames"] += 1
            return

        self.metrics["valid_pose_frames"] += 1

        shoulder = np.array([
            (landmarks[11].x + landmarks[12].x) / 2,
            (landmarks[11].y + landmarks[12].y) / 2,
        ])

        hip = np.array([
            (landmarks[23].x + landmarks[24].x) / 2,
            (landmarks[23].y + landmarks[24].y) / 2,
        ])

        torso_length = (
            np.linalg.norm(shoulder - hip) + 1e-6
        )

        coords = np.array([
            [landmarks[i].x, landmarks[i].y, landmarks[i].z]
            for i in required
        ])

        self.spatial_history.append(coords)
        self.time_history.append(now)

        if len(self.spatial_history) < 3:
            return

        arr = np.asarray(self.spatial_history)

        movement_variance = float(
            np.var(arr, axis=0).sum()
            / torso_length
        )

        dt = max(
            0.001,
            self.time_history[-1]
            - self.time_history[-2],
        )

        previous_dt = max(
            0.001,
            self.time_history[-2]
            - self.time_history[-3],
        )

        velocity_vector = (
            arr[-1] - arr[-2]
        ) / dt

        previous_velocity_vector = (
            arr[-2] - arr[-3]
        ) / previous_dt

        velocity = float(
            np.linalg.norm(velocity_vector)
            / torso_length
        )

        previous_velocity = float(
            np.linalg.norm(previous_velocity_vector)
            / torso_length
        )

        acceleration = float(
            abs(velocity - previous_velocity) / dt
        )

        self.metrics["velocity"].append(velocity)
        self.metrics["acceleration"].append(acceleration)
        self.metrics["movement_variance"].append(
            movement_variance
        )

    def _process_face(self, landmarks):
        self.metrics["face_frames"] += 1

        # Left/right eye landmarks.
        # These are normalized geometric measurements.
        top = np.array([
            landmarks[159].x,
            landmarks[159].y,
        ])

        bottom = np.array([
            landmarks[145].x,
            landmarks[145].y,
        ])

        inner = np.array([
            landmarks[133].x,
            landmarks[133].y,
        ])

        outer = np.array([
            landmarks[33].x,
            landmarks[33].y,
        ])

        eye_height = np.linalg.norm(
            top - bottom
        )

        eye_width = (
            np.linalg.norm(inner - outer)
            + 1e-6
        )

        ear = eye_height / eye_width

        currently_blinking = ear < 0.20

        # Count only the transition into a blink.
        if (
            currently_blinking
            and not self.last_blink_state
        ):
            self.metrics["blink_count"] += 1

        self.last_blink_state = currently_blinking

        # MediaPipe iris landmark.
        iris = np.array([
            landmarks[468].x,
            landmarks[468].y,
        ])

        eye_center = (
            inner + outer
        ) / 2.0

        gaze_distance = np.linalg.norm(
            iris - eye_center
        )

        gaze_deviation = (
            gaze_distance > eye_width * 0.15
        )

        # Count transition rather than every video frame.
        if (
            gaze_deviation
            and not self.last_gaze_deviation
        ):
            self.metrics[
                "gaze_deviation_count"
            ] += 1

        self.last_gaze_deviation = gaze_deviation

    def snapshot(self):
        with self.lock:
            return {
                "running": self.running,
                "start_time": self.start_time,
                "recording_path": self.recording_path,
                "metrics": {
                    key: list(value)
                    if isinstance(value, list)
                    else value
                    for key, value in self.metrics.items()
                },
            }

    def recv(self, frame: av.VideoFrame):
        img = frame.to_ndarray(format="bgr24")

        # Standardize recording resolution.
        img = cv2.resize(img, (640, 480))

        now = time.perf_counter()

        with self.lock:
            running = self.running
            writer = self.writer
            start = self.start_time

        if not running:
            cv2.putText(
                img,
                "CAMERA READY",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            return av.VideoFrame.from_ndarray(
                img,
                format="bgr24",
            )

        elapsed = (
            now - start
            if start is not None
            else 0
        )

        timestamp_ms = int(now * 1000)

        if self.last_timestamp_ms is not None:
            timestamp_ms = max(
                timestamp_ms,
                self.last_timestamp_ms + 1,
            )

        self.last_timestamp_ms = timestamp_ms

        rgb = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB,
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb,
        )

        pose_result = (
            self.pose_detector.detect_for_video(
                mp_image,
                timestamp_ms,
            )
        )

        if pose_result.pose_landmarks:
            self._process_pose(
                pose_result.pose_landmarks[0],
                now,
            )

        face_result = (
            self.face_detector.detect_for_video(
                mp_image,
                timestamp_ms,
            )
        )

        if face_result.face_landmarks:
            self._process_face(
                face_result.face_landmarks[0]
            )

        # Camera-only UI.
        cv2.rectangle(
            img,
            (0, 0),
            (640, 54),
            (20, 20, 20),
            -1,
        )

        cv2.putText(
            img,
            "CAMERA / BIOMARKER RECORDING",
            (15, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            img,
            f"Recording: {elapsed:.1f}s",
            (15, 46),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (210, 210, 210),
            1,
        )

        # Small pose landmarks for researcher visibility.
        if pose_result.pose_landmarks:
            p = pose_result.pose_landmarks[0]

            for idx in [0, 11, 12, 23, 24]:
                x = int(p[idx].x * 640)
                y = int(p[idx].y * 480)

                cv2.circle(
                    img,
                    (x, y),
                    3,
                    (0, 220, 220),
                    -1,
                )

        # Save the exact camera frame.
        if writer is not None:
            writer.write(img)

        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24",
        )


# ============================================================
# TEST SESSION STATE
# ============================================================
class FocusMissionEngine:
    """
    Runs the cognitive task independently from the camera window.

    This is intentionally separate from the camera processor.
    """

    def __init__(self):
        self.lock = threading.RLock()

        self.running = False
        self.start_time = None
        self.current_trial = None
        self.next_trial_time = None

        self.trials = []

        self.response_queue = deque(maxlen=20)

    def start(self):
        with self.lock:
            self.running = True
            self.start_time = time.perf_counter()
            self.current_trial = None
            self.next_trial_time = (
                self.start_time + 2.0
            )
            self.trials = []
            self.response_queue.clear()

    def stop(self):
        with self.lock:
            self.running = False

    def elapsed(self):
        with self.lock:
            if self.start_time is None:
                return 0.0

            return (
                time.perf_counter()
                - self.start_time
            )

    def get_phase(self):
        elapsed = self.elapsed()
        cursor = 0.0

        for phase in PHASES:
            if elapsed < cursor + phase["duration"]:
                phase_copy = dict(phase)
                phase_copy["elapsed"] = (
                    elapsed - cursor
                )
                phase_copy["remaining"] = (
                    cursor
                    + phase["duration"]
                    - elapsed
                )
                return phase_copy

            cursor += phase["duration"]

        return None

    def _stimulus_for_phase(self, phase):
        target = (
            random.random()
            < phase["target_probability"]
        )

        rule = phase["rule"]

        if rule == "circle":
            stimulus = "circle"

        elif rule == "shape":
            if target:
                stimulus = "circle"
            else:
                stimulus = random.choice([
                    "square",
                    "triangle",
                    "diamond",
                ])

        elif rule == "go_nogo":
            stimulus = (
                "green_circle"
                if target
                else "red_circle"
            )

        else:
            if target:
                stimulus = "circle"
            else:
                stimulus = random.choice([
                    "square",
                    "triangle",
                    "diamond",
                    "cross",
                ])

        return Trial(
            phase=phase["name"],
            target=target,
            stimulus=stimulus,
            onset=time.perf_counter(),
        )

    def maybe_generate_trial(self):
        with self.lock:
            if not self.running:
                return

            now = time.perf_counter()
            phase = self.get_phase()

            if phase is None:
                self.running = False
                return

            current = self.current_trial

            if current is not None:
                finished = (
                    current.response is not None
                    or current.outcome
                    in [
                        "omission",
                        "correct_rejection",
                        "commission",
                        "hit",
                    ]
                )

                # Response window.
                if (
                    current.response is None
                    and now - current.onset > 1.50
                ):
                    current.outcome = (
                        "omission"
                        if current.target
                        else "correct_rejection"
                    )

                if not finished:
                    return

            if now >= self.next_trial_time:
                self.current_trial = (
                    self._stimulus_for_phase(
                        phase
                    )
                )

                self.trials.append(
                    self.current_trial
                )

                self.next_trial_time = (
                    now
                    + phase["stimulus_interval"]
                    + random.uniform(
                        0.10,
                        0.40,
                    )
                )

    def register_response(self):
        with self.lock:
            if not self.running:
                return

            self.response_queue.append(
                time.perf_counter()
            )

    def process_response_queue(self):
        with self.lock:
            if not self.response_queue:
                return

            responses = list(
                self.response_queue
            )

            self.response_queue.clear()

            trial = self.current_trial

            if trial is None:
                return

            if trial.response is not None:
                return

            response_time = responses[-1]

            rt = (
                response_time
                - trial.onset
            )

            if not 0.05 <= rt <= 1.50:
                return

            trial.response = response_time
            trial.reaction_time = rt

            if trial.target:
                trial.outcome = "hit"
            else:
                trial.outcome = "commission"

    def tick(self):
        if not self.running:
            return

        self.process_response_queue()
        self.maybe_generate_trial()

    def current_stimulus(self):
        with self.lock:
            if self.current_trial is None:
                return None

            now = time.perf_counter()
            trial = self.current_trial

            if (
                trial.response is not None
                or trial.outcome
                in [
                    "omission",
                    "correct_rejection",
                    "commission",
                    "hit",
                ]
            ):
                return None

            if now - trial.onset > 1.50:
                return None

            return trial.stimulus

    def snapshot(self):
        with self.lock:
            return {
                "running": self.running,
                "start_time": self.start_time,
                "trials": [
                    t.to_dict()
                    for t in self.trials
                ],
            }


# ============================================================
# CREATE OBJECTS IN SESSION STATE
# ============================================================
if "camera_processor_reference" not in st.session_state:
    st.session_state[
        "camera_processor_reference"
    ] = None

if "mission_engine" not in st.session_state:
    st.session_state[
        "mission_engine"
    ] = FocusMissionEngine()


# ============================================================
# RESULT / BIOMARKER CALCULATION
# ============================================================
def calculate_assessment_results(
    participant_id,
    name,
    gender,
    age,
    camera_snapshot,
    mission_snapshot,
):
    trials = mission_snapshot["trials"]

    hits = sum(
        t["outcome"] == "hit"
        for t in trials
    )

    omissions = sum(
        t["outcome"] == "omission"
        for t in trials
    )

    commissions = sum(
        t["outcome"] == "commission"
        for t in trials
    )

    correct_rejections = sum(
        t["outcome"] == "correct_rejection"
        for t in trials
    )

    reaction_times = [
        t["reaction_time"]
        for t in trials
        if t["reaction_time"] is not None
    ]

    cm = camera_snapshot["metrics"]

    valid_frames = cm["valid_pose_frames"]
    invalid_frames = cm["invalid_pose_frames"]

    total_pose_frames = (
        valid_frames
        + invalid_frames
    )

    # Descriptive biomarker summary.
    # These values are stored for subsequent ADFI processing.
    biomarkers = {
        "mean_velocity": mean(
            cm["velocity"]
        ),
        "mean_acceleration": mean(
            cm["acceleration"]
        ),
        "movement_variance": mean(
            cm["movement_variance"]
        ),
        "blink_count": cm["blink_count"],
        "gaze_deviation_count": cm[
            "gaze_deviation_count"
        ],
        "mean_reaction_time": mean(
            reaction_times
        ),
        "reaction_time_variability": std(
            reaction_times
        ),
        "omissions": omissions,
        "commissions": commissions,
        "timeouts": omissions,
        "hits": hits,
        "correct_rejections": correct_rejections,
        "valid_pose_frame_ratio": ratio(
            valid_frames,
            total_pose_frames,
        ),
    }

    # Keep ADFI fields explicit instead of inventing
    # coefficients or diagnostic thresholds.
    adfi = {
        "S_beh": None,
        "S_perf": None,
        "ADFI": None,
        "status": (
            "Pending validated coefficient model"
        ),
    }

    return {
        "session_id": st.session_state[
            "session_id"
        ],
        "participant": {
            "id": participant_id,
            "name": name,
            "gender": gender,
            "age": int(age),
        },
        "assessment": {
            "name": (
                "Focus Mission Multimodal "
                "Attention Assessment"
            ),
            "type": (
                "Computerized Continuous "
                "Performance / Go-No-Go "
                "research assessment"
            ),
            "duration_seconds": TOTAL_DURATION,
            "completed_at": utc_now(),
        },
        "performance": {
            "total_trials": len(trials),
            "hits": hits,
            "omissions": omissions,
            "commissions": commissions,
            "correct_rejections": (
                correct_rejections
            ),
            "mean_reaction_time": mean(
                reaction_times
            ),
            "reaction_time_sd": std(
                reaction_times
            ),
        },
        "biomarkers": biomarkers,
        "adfi": adfi,
        "camera_quality": {
            "valid_pose_frames": valid_frames,
            "invalid_pose_frames": invalid_frames,
            "valid_pose_ratio": ratio(
                valid_frames,
                total_pose_frames,
            ),
            "face_frames": cm[
                "face_frames"
            ],
        },
        "recording": {
            "local_path": camera_snapshot[
                "recording_path"
            ],
        },
        "trials": trials,
    }


# ============================================================
# FIRESTORE SAVE
# ============================================================
def save_results_to_firestore(results):
    if db is None:
        return False, (
            "Firebase is not configured. "
            "Add firebase_credentials.json."
        )

    participant_id = sanitize_id(
        results["participant"]["id"]
    )

    session_id = results["session_id"]

    try:
        participant_ref = db.collection(
            "participants"
        ).document(participant_id)

        # Primary participant record.
        participant_ref.set(
            {
                "participant_id": participant_id,
                "name": results[
                    "participant"
                ]["name"],
                "gender": results[
                    "participant"
                ]["gender"],
                "age": results[
                    "participant"
                ]["age"],
                "updated_at": firestore.SERVER_TIMESTAMP,
            },
            merge=True,
        )

        # Full assessment session.
        participant_ref.collection(
            "assessments"
        ).document(session_id).set(
            results
        )

        # Convenient latest-assessment summary.
        participant_ref.set(
            {
                "latest_assessment": {
                    "session_id": session_id,
                    "completed_at": results[
                        "assessment"
                    ]["completed_at"],
                    "biomarkers": results[
                        "biomarkers"
                    ],
                    "performance": results[
                        "performance"
                    ],
                    "adfi": results[
                        "adfi"
                    ],
                }
            },
            merge=True,
        )

        return True, "Assessment saved successfully."

    except Exception as exc:
        return False, f"Firestore save failed: {exc}"


def participant_exists(participant_id):
    if db is None:
        return False

    try:
        ref = db.collection(
            "participants"
        ).document(
            sanitize_id(participant_id)
        )

        return ref.get().exists

    except Exception:
        return False


# ============================================================
# CSS — PATIENT-FACING DESIGN
# ============================================================
st.markdown(
    """
    <style>
    .block-container {
        max-width: 1380px;
        padding-top: 1.1rem;
        padding-bottom: 3rem;
    }

    .hero {
        padding: 28px;
        border-radius: 24px;
        border: 1px solid rgba(128,128,128,.22);
        background: linear-gradient(
            135deg,
            rgba(128,128,128,.10),
            rgba(128,128,128,.035)
        );
        margin-bottom: 18px;
    }

    .hero h1 {
        margin: 0;
        font-size: 2.45rem;
    }

    .hero p {
        margin-top: 8px;
        opacity: .76;
    }

    .mission-card {
        padding: 22px;
        border-radius: 20px;
        border: 1px solid rgba(128,128,128,.20);
        background: rgba(128,128,128,.055);
        margin-bottom: 15px;
    }

    .mission-title {
        font-size: 1.25rem;
        font-weight: 700;
    }

    .stimulus-box {
        min-height: 390px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 24px;
        border: 1px solid rgba(128,128,128,.20);
        background:
            radial-gradient(
                circle at center,
                rgba(128,128,128,.10),
                rgba(128,128,128,.025) 45%,
                transparent 70%
            );
    }

    .target {
        width: 155px;
        height: 155px;
        border-radius: 50%;
        background: #42d86f;
        border: 8px solid rgba(255,255,255,.88);
        box-shadow: 0 0 0 14px rgba(66,216,111,.13);
    }

    .nogreen {
        width: 155px;
        height: 155px;
        border-radius: 50%;
        background: #e05a5a;
        border: 8px solid rgba(255,255,255,.88);
    }

    .square {
        width: 155px;
        height: 155px;
        background: #a8a8a8;
        border: 8px solid rgba(255,255,255,.75);
    }

    .triangle {
        width: 0;
        height: 0;
        border-left: 90px solid transparent;
        border-right: 90px solid transparent;
        border-bottom: 165px solid #a8a8a8;
    }

    .diamond {
        width: 145px;
        height: 145px;
        background: #a8a8a8;
        transform: rotate(45deg);
        border: 8px solid rgba(255,255,255,.70);
    }

    .waiting {
        font-size: 1.25rem;
        opacity: .5;
    }

    .rule-card {
        text-align: center;
        padding: 13px;
        border-radius: 15px;
        background: rgba(128,128,128,.07);
        border: 1px solid rgba(128,128,128,.16);
    }

    .big-button button {
        min-height: 74px !important;
        font-size: 1.15rem !important;
        font-weight: 800 !important;
        border-radius: 17px !important;
    }

    .success-card {
        padding: 28px;
        border-radius: 22px;
        border: 1px solid rgba(66,216,111,.30);
        background: rgba(66,216,111,.07);
    }

    .privacy-note {
        font-size: .82rem;
        opacity: .62;
        line-height: 1.5;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================
st.markdown(
    """
    <div class="hero">
        <h1>🎯 Focus Mission</h1>
        <p>
        A short, progressive attention mission combining
        computerized response tasks with camera-based
        behavioral and oculomotor measurements.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SETUP SCREEN
# ============================================================
if not st.session_state["assessment_started"]:

    st.subheader("Participant Registration")

    c1, c2 = st.columns(2)

    with c1:
        name = st.text_input(
            "Full Name",
            placeholder="Enter participant name",
        )

        participant_id = st.text_input(
            "Participant ID *",
            placeholder="Example: P001",
            help=(
                "This becomes the primary key / "
                "Firestore document ID."
            ),
        )

        age = st.number_input(
            "Age",
            min_value=3,
            max_value=100,
            value=18,
            step=1,
        )

    with c2:
        gender = st.selectbox(
            "Gender",
            [
                "Prefer not to say",
                "Male",
                "Female",
                "Other",
            ],
        )

        st.markdown(
            """
            <div class="mission-card">
                <div class="mission-title">
                    🧠 What happens?
                </div>
                <p>
                You will complete four short missions.
                Each mission becomes a little more challenging.
                </p>
                <p>
                Watch the separate Focus Mission screen and
                press <b>RESPOND</b> when the target appears.
                </p>
                <p>
                The camera records movement and eye-related
                measurements in the background.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    consent = st.checkbox(
        "I understand the instructions and agree to participate."
    )

    st.markdown(
        """
        <div class="privacy-note">
        Research note: camera-derived measurements and assessment
        results should be handled according to your institution's
        consent, privacy, retention, and data-security requirements.
        This system is not a stand-alone ADHD diagnosis.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button(
        "🚀 Begin Focus Mission",
        type="primary",
        use_container_width=True,
    ):
        pid = sanitize_id(participant_id)

        if not name.strip():
            st.error("Please enter the participant name.")

        elif not pid:
            st.error("Please enter the Participant ID.")

        elif not consent:
            st.error(
                "Please confirm participation before starting."
            )

        elif db is not None and participant_exists(pid):
            st.error(
                "This Participant ID already exists. "
                "Use a unique ID for a new participant."
            )

        else:
            session_id = (
                f"{pid}_"
                f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )

            st.session_state[
                "participant_id"
            ] = pid

            st.session_state["name"] = name.strip()
            st.session_state["gender"] = gender
            st.session_state["age"] = int(age)

            st.session_state[
                "session_id"
            ] = session_id

            st.session_state[
                "assessment_started"
            ] = True

            st.session_state[
                "assessment_finished"
            ] = False

            st.session_state[
                "saved_to_firebase"
            ] = False

            st.rerun()


# ============================================================
# ASSESSMENT SCREEN
# ============================================================
if st.session_state["assessment_started"]:

    mission = st.session_state[
        "mission_engine"
    ]

    # --------------------------------------------------------
    # CAMERA WINDOW — LEFT
    # --------------------------------------------------------
    camera_col, test_col = st.columns(
        [1.05, 1.75],
        gap="large",
    )

    with camera_col:
        st.markdown(
            "### 📹 Camera & Biomarkers"
        )

        st.caption(
            "Your camera is monitored separately "
            "from the Focus Mission."
        )

        rtc_config = RTCConfiguration(
            {
                "iceServers": [
                    {
                        "urls": [
                            "stun:stun.l.google.com:19302"
                        ]
                    }
                ]
            }
        )

        ctx = webrtc_streamer(
            key=(
                "camera_"
                + st.session_state["session_id"]
            ),
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=rtc_config,
            video_processor_factory=(
                CameraBiomarkerProcessor
            ),
            media_stream_constraints={
                "video": {
                    "width": {
                        "ideal": 640
                    },
                    "height": {
                        "ideal": 480
                    },
                    "frameRate": {
                        "ideal": 20
                    },
                },
                "audio": False,
            },
            async_processing=True,
        )

        if ctx and ctx.video_processor:

            processor = (
                ctx.video_processor
            )

            if (
                processor.start_time
                is None
            ):
                recording_dir = "recordings"
                os.makedirs(
                    recording_dir,
                    exist_ok=True,
                )

                recording_path = os.path.join(
                    recording_dir,
                    (
                        st.session_state[
                            "session_id"
                        ]
                        + ".mp4"
                    ),
                )

                processor.start_assessment(
                    recording_path
                )

            st.session_state[
                "camera_processor_reference"
            ] = processor

        st.info(
            "Sit comfortably. Keep your face and "
            "upper body visible. Try to behave naturally."
        )

    # --------------------------------------------------------
    # COGNITIVE TEST WINDOW — RIGHT
    # --------------------------------------------------------
    with test_col:

        if not mission.running:
            mission.start()

        mission.tick()

        phase = mission.get_phase()

        if phase is None:
            mission.stop()
            st.session_state[
                "assessment_finished"
            ] = True

        else:
            st.markdown(
                f"### 🛰️ Mission {phase['id']} / 4"
            )

            progress_before = (
                TOTAL_DURATION
                - phase["remaining"]
            )

            st.progress(
                min(
                    1.0,
                    progress_before
                    / TOTAL_DURATION,
                )
            )

            p1, p2, p3 = st.columns(3)

            with p1:
                st.metric(
                    "Mission",
                    phase["name"],
                )

            with p2:
                st.metric(
                    "Time Left",
                    f"{int(phase['remaining'])}s",
                )

            with p3:
                st.metric(
                    "Trials",
                    len(mission.trials),
                )

            st.markdown(
                f"""
                <div class="mission-card">
                    <div class="mission-title">
                        {phase["title"]}
                    </div>
                    <p>
                        {phase["description"]}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            stimulus = (
                mission.current_stimulus()
            )

            # Separate visual test area.
            if stimulus == "circle":
                html = (
                    '<div class="stimulus-box">'
                    '<div class="target"></div>'
                    '</div>'
                )

            elif stimulus == "green_circle":
                html = (
                    '<div class="stimulus-box">'
                    '<div class="target"></div>'
                    '</div>'
                )

            elif stimulus == "red_circle":
                html = (
                    '<div class="stimulus-box">'
                    '<div class="nogreen"></div>'
                    '</div>'
                )

            elif stimulus == "square":
                html = (
                    '<div class="stimulus-box">'
                    '<div class="square"></div>'
                    '</div>'
                )

            elif stimulus == "triangle":
                html = (
                    '<div class="stimulus-box">'
                    '<div class="triangle"></div>'
                    '</div>'
                )

            elif stimulus == "diamond":
                html = (
                    '<div class="stimulus-box">'
                    '<div class="diamond"></div>'
                    '</div>'
                )

            elif stimulus == "cross":
                html = (
                    '<div class="stimulus-box">'
                    '<div style="font-size:150px;'
                    'font-weight:900;">×</div>'
                    '</div>'
                )

            else:
                html = (
                    '<div class="stimulus-box">'
                    '<div class="waiting">'
                    'Get ready…'
                    '</div>'
                    '</div>'
                )

            st.markdown(
                html,
                unsafe_allow_html=True,
            )

            st.markdown(
                '<div class="big-button">',
                unsafe_allow_html=True,
            )

            if st.button(
                "🟢  RESPOND",
                type="primary",
                use_container_width=True,
                key=(
                    "respond_"
                    + st.session_state[
                        "session_id"
                    ]
                    + "_"
                    + str(
                        len(mission.trials)
                    )
                ),
            ):
                mission.register_response()

            st.markdown(
                "</div>",
                unsafe_allow_html=True,
            )

            st.markdown(
                """
                <div class="rule-card">
                    <b>Remember</b><br>
                    Respond only when you see the target.
                    If it is not the target, do nothing.
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Refresh the Streamlit page to animate trials.
            # The camera processor continues independently.
            time.sleep(0.06)
            st.rerun()


# ============================================================
# FINISH / SAVE
# ============================================================
if (
    st.session_state["assessment_started"]
    and st.session_state["assessment_finished"]
):

    mission = st.session_state[
        "mission_engine"
    ]

    processor = st.session_state[
        "camera_processor_reference"
    ]

    if processor is not None:
        processor.stop_assessment()

        camera_snapshot = (
            processor.snapshot()
        )

    else:
        camera_snapshot = {
            "recording_path": None,
            "metrics": {
                "velocity": [],
                "acceleration": [],
                "movement_variance": [],
                "blink_count": 0,
                "gaze_deviation_count": 0,
                "valid_pose_frames": 0,
                "invalid_pose_frames": 0,
                "face_frames": 0,
            },
        }

    mission_snapshot = (
        mission.snapshot()
    )

    results = calculate_assessment_results(
        st.session_state[
            "participant_id"
        ],
        st.session_state["name"],
        st.session_state["gender"],
        st.session_state["age"],
        camera_snapshot,
        mission_snapshot,
    )

    st.session_state[
        "last_results"
    ] = results

    st.markdown(
        """
        <div class="success-card">
            <h2>🎉 Mission Complete</h2>
            <p>
            The assessment has finished successfully.
            Your responses and camera-derived measurements
            have been collected.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.progress(1.0)

    st.subheader("Assessment Summary")

    perf = results["performance"]
    bio = results["biomarkers"]

    a, b, c, d = st.columns(4)

    a.metric(
        "Total Trials",
        perf["total_trials"],
    )

    b.metric(
        "Hits",
        perf["hits"],
    )

    c.metric(
        "Omissions",
        perf["omissions"],
    )

    d.metric(
        "Commissions",
        perf["commissions"],
    )

    st.subheader("Recorded Biomarkers")

    b1, b2, b3, b4 = st.columns(4)

    b1.metric(
        "Mean RT",
        (
            f'{bio["mean_reaction_time"]:.3f}s'
            if bio["mean_reaction_time"]
            is not None
            else "—"
        ),
    )

    b2.metric(
        "RT Variability",
        (
            f'{bio["reaction_time_variability"]:.3f}s'
            if bio[
                "reaction_time_variability"
            ] is not None
            else "—"
        ),
    )

    b3.metric(
        "Blink Count",
        bio["blink_count"],
    )

    b4.metric(
        "Gaze Deviations",
        bio[
            "gaze_deviation_count"
        ],
    )

    st.subheader("Behavioral Activity")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Mean Velocity",
        (
            f'{bio["mean_velocity"]:.5f}'
            if bio["mean_velocity"]
            is not None
            else "—"
        ),
    )

    c2.metric(
        "Mean Acceleration",
        (
            f'{bio["mean_acceleration"]:.5f}'
            if bio["mean_acceleration"]
            is not None
            else "—"
        ),
    )

    c3.metric(
        "Movement Variance",
        (
            f'{bio["movement_variance"]:.5f}'
            if bio["movement_variance"]
            is not None
            else "—"
        ),
    )

    st.subheader("Database")

    if not st.session_state[
        "saved_to_firebase"
    ]:

        if st.button(
            "💾 Save Participant + Biomarkers to Firebase",
            type="primary",
            use_container_width=True,
        ):
            ok, message = (
                save_results_to_firestore(
                    results
                )
            )

            if ok:
                st.session_state[
                    "saved_to_firebase"
                ] = True

                st.success(message)

            else:
                st.error(message)

    else:
        st.success(
            "✓ Participant information and assessment "
            "biomarkers are saved in Firestore."
        )

    st.subheader("Export")

    json_bytes = json.dumps(
        results,
        indent=2,
        default=str,
    ).encode("utf-8")

    st.download_button(
        "⬇️ Download Complete Assessment JSON",
        data=json_bytes,
        file_name=(
            st.session_state[
                "participant_id"
            ]
            + "_assessment.json"
        ),
        mime="application/json",
        use_container_width=True,
    )

    st.info(
        "The ADFI fields are intentionally marked as pending "
        "until validated coefficients are trained from labelled "
        "data. This prevents the system from producing a "
        "fabricated clinical score."
    )


# ============================================================
# CLINICIAN FOOTER
# ============================================================
st.divider()

st.caption(
    "Focus Mission Research Prototype • "
    "Camera biomarkers + computerized attention performance • "
    "Not a stand-alone ADHD diagnosis"
)
