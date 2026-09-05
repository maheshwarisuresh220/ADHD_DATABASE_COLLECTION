import os
import time
import random
import socket
import threading
import json
import re
from collections import deque
from dataclasses import dataclass, asdict

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import urllib.request

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# FOCUS MISSION — MULTIMODAL ATTENTION ASSESSMENT
# ============================================================
# IMPORTANT:
# This application is an assessment/research prototype.
# It must NOT be presented as a stand-alone ADHD diagnosis.
#
# Main improvements:
# 1. Real trial-based CPT/Go-No-Go logic instead of elapsed-second visuals.
# 2. Literacy-neutral geometric stimuli.
# 3. Training -> Focus -> Inhibition -> Distraction progression.
# 4. Thread-safe response handling between Streamlit and WebRTC.
# 5. Real omission/commission/hit/correct-rejection calculations.
# 6. Real reaction-time statistics.
# 7. Pose-quality monitoring.
# 8. No fabricated/mock clinical results.
# 9. Clinician-only results section.
# 10. JSON export for later ADFI/Firebase processing.


# ============================================================
# PAGE / CONSTANTS
# ============================================================
st.set_page_config(
    page_title="Focus Mission Assessment",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MODEL_PATH = "pose_landmarker_lite.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)

# Keep the first version practical for pilot data collection.
# Total = 7 minutes.
PHASES = [
    {
        "name": "TRAINING",
        "duration": 60,
        "interval": 2.8,
        "target_probability": 1.00,
        "rule": "circle",
    },
    {
        "name": "FOCUS",
        "duration": 120,
        "interval": 1.8,
        "target_probability": 0.50,
        "rule": "circle",
    },
    {
        "name": "CONTROL",
        "duration": 120,
        "interval": 1.35,
        "target_probability": 0.50,
        "rule": "go_nogo",
    },
    {
        "name": "DISTRACTION",
        "duration": 120,
        "interval": 1.05,
        "target_probability": 0.45,
        "rule": "shape",
    },
]

TOTAL_DURATION = sum(p["duration"] for p in PHASES)


# ============================================================
# HELPERS
# ============================================================
def ensure_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Preparing attention-tracking model..."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def mean_or_none(values):
    return float(np.mean(values)) if values else None


def std_or_none(values):
    return float(np.std(values)) if values else None


def safe_ratio(a, b):
    return float(a / b) if b else 0.0


ensure_model()


# ============================================================
# SESSION STATE
# ============================================================
DEFAULTS = {
    "participant_id": "",
    "protocol_started": False,
    "protocol_finished": False,
    "protocol_start_wall": None,
    "export_payload": None,
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# TRIAL MODEL
# ============================================================
@dataclass
class Trial:
    phase: str
    target: bool
    stimulus: str
    onset: float
    response: float | None = None
    reaction_time: float | None = None
    outcome: str = "pending"


# ============================================================
# WEBRTC PROCESSOR
# ============================================================
class MCPSProcessor:
    """
    Real-time assessment processor.

    The WebRTC thread owns the camera processing.
    Streamlit's response button communicates with this processor
    through register_response(). Access is protected by a lock.

    The processor deliberately does not calculate an ADHD diagnosis.
    It only collects measurable behavioral/performance features.
    """

    def __init__(self):
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)

        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            min_pose_detection_confidence=0.60,
            min_pose_presence_confidence=0.60,
        )

        self.detector = vision.PoseLandmarker.create_from_options(options)

        self.lock = threading.RLock()

        self.running = False
        self.protocol_start = None
        self.last_timestamp_ms = None

        self.position_valid = False
        self.current_phase = "WAITING"

        self.current_trial = None
        self.trials = []
        self.response_queue = deque(maxlen=32)

        self.next_trial_time = None

        # Movement history.
        self.spatial_history = deque(maxlen=15)
        self.time_history = deque(maxlen=15)

        # Raw measurements.
        self.metrics = {
            "velocity": [],
            "acceleration": [],
            "movement_variance": [],
            "valid_frames": 0,
            "invalid_frames": 0,
        }

    # --------------------------------------------------------
    # Protocol control
    # --------------------------------------------------------
    def start_protocol(self):
        now = time.perf_counter()

        with self.lock:
            self.running = True
            self.protocol_start = now
            self.last_timestamp_ms = None
            self.position_valid = False
            self.current_phase = "TRAINING"

            self.current_trial = None
            self.trials.clear()
            self.response_queue.clear()

            self.next_trial_time = now + 2.0

            self.spatial_history.clear()
            self.time_history.clear()

            self.metrics = {
                "velocity": [],
                "acceleration": [],
                "movement_variance": [],
                "valid_frames": 0,
                "invalid_frames": 0,
            }

    def stop_protocol(self):
        with self.lock:
            self.running = False

    # --------------------------------------------------------
    # Response input
    # --------------------------------------------------------
    def register_response(self):
        """
        Called by the large patient response button.

        We record a monotonic server-side timestamp.
        """
        with self.lock:
            self.response_queue.append(time.perf_counter())

    # --------------------------------------------------------
    # Phase logic
    # --------------------------------------------------------
    def get_phase(self, elapsed):
        cursor = 0.0

        for phase in PHASES:
            end = cursor + phase["duration"]

            if elapsed < end:
                phase_copy = dict(phase)
                phase_copy["phase_elapsed"] = elapsed - cursor
                phase_copy["phase_remaining"] = end - elapsed
                return phase_copy

            cursor = end

        return None

    # --------------------------------------------------------
    # Trial generation
    # --------------------------------------------------------
    def generate_trial(self, phase, now):
        target = random.random() < phase["target_probability"]
        rule = phase["rule"]

        if rule in ("circle", "go_nogo"):
            stimulus = "circle" if target else "no_go"
        else:
            # Target remains the circle. Distractors are geometric.
            if target:
                stimulus = "circle"
            else:
                stimulus = random.choice(["square", "triangle", "diamond"])

        trial = Trial(
            phase=phase["name"],
            target=target,
            stimulus=stimulus,
            onset=now,
        )

        with self.lock:
            self.current_trial = trial
            self.trials.append(trial)

            self.next_trial_time = (
                now
                + phase["interval"]
                + random.uniform(0.10, 0.45)
            )

    # --------------------------------------------------------
    # Response matching
    # --------------------------------------------------------
    def process_responses(self, now):
        with self.lock:
            responses = list(self.response_queue)
            self.response_queue.clear()
            trial = self.current_trial

        if not responses or trial is None:
            return

        # Do not let multiple button clicks create multiple responses.
        if trial.response is not None:
            return

        response_time = responses[-1]
        reaction_time = response_time - trial.onset

        # Ignore clicks that clearly occurred outside the active trial.
        if 0.05 <= reaction_time <= 1.50:
            trial.response = response_time
            trial.reaction_time = reaction_time
            trial.outcome = "hit" if trial.target else "commission"

            with self.lock:
                self.current_trial = trial

    def expire_trial(self, now):
        with self.lock:
            trial = self.current_trial

        if trial is None:
            return

        # Stimulus response window.
        if trial.response is None and now - trial.onset > 1.50:
            trial.outcome = (
                "omission" if trial.target else "correct_rejection"
            )

            with self.lock:
                self.current_trial = trial

    # --------------------------------------------------------
    # Pose processing
    # --------------------------------------------------------
    def process_pose(self, landmarks, timestamp):
        required = [0, 11, 12, 23, 24]

        visible = True

        for idx in required:
            point = landmarks[idx]

            if hasattr(point, "visibility"):
                if point.visibility < 0.55:
                    visible = False
                    break

        self.position_valid = visible

        if not visible:
            self.metrics["invalid_frames"] += 1
            return

        self.metrics["valid_frames"] += 1

        shoulder = np.array([
            (landmarks[11].x + landmarks[12].x) / 2.0,
            (landmarks[11].y + landmarks[12].y) / 2.0,
        ])

        hip = np.array([
            (landmarks[23].x + landmarks[24].x) / 2.0,
            (landmarks[23].y + landmarks[24].y) / 2.0,
        ])

        torso_length = np.linalg.norm(shoulder - hip) + 1e-6

        coords = np.array([
            [landmarks[i].x, landmarks[i].y, landmarks[i].z]
            for i in required
        ])

        self.spatial_history.append(coords)
        self.time_history.append(timestamp)

        if len(self.spatial_history) < 3:
            return

        arr = np.asarray(self.spatial_history)

        variance = float(
            np.var(arr, axis=0).sum() / torso_length
        )

        dt = max(
            1e-3,
            self.time_history[-1] - self.time_history[-2],
        )

        previous_dt = max(
            1e-3,
            self.time_history[-2] - self.time_history[-3],
        )

        velocity_vector = (arr[-1] - arr[-2]) / dt
        previous_velocity_vector = (arr[-2] - arr[-3]) / previous_dt

        velocity = float(
            np.linalg.norm(velocity_vector) / torso_length
        )

        previous_velocity = float(
            np.linalg.norm(previous_velocity_vector) / torso_length
        )

        acceleration = float(
            abs(velocity - previous_velocity) / dt
        )

        self.metrics["velocity"].append(velocity)
        self.metrics["acceleration"].append(acceleration)
        self.metrics["movement_variance"].append(variance)

    # --------------------------------------------------------
    # Drawing
    # --------------------------------------------------------
    def draw_stimulus(self, img, stimulus):
        h, w = img.shape[:2]
        cx, cy = w // 2, h // 2
        size = max(45, min(w, h) // 7)

        # Neutral grey distractors.
        if stimulus == "circle":
            cv2.circle(img, (cx, cy), size, (70, 220, 90), -1)

        elif stimulus == "square":
            cv2.rectangle(
                img,
                (cx - size, cy - size),
                (cx + size, cy + size),
                (185, 185, 185),
                -1,
            )

        elif stimulus == "triangle":
            pts = np.array(
                [
                    [cx, cy - size],
                    [cx - size, cy + size],
                    [cx + size, cy + size],
                ],
                np.int32,
            )
            cv2.fillPoly(img, [pts], (185, 185, 185))

        elif stimulus == "diamond":
            pts = np.array(
                [
                    [cx, cy - size],
                    [cx - size, cy],
                    [cx, cy + size],
                    [cx + size, cy],
                ],
                np.int32,
            )
            cv2.fillPoly(img, [pts], (185, 185, 185))

        elif stimulus == "no_go":
            cv2.circle(img, (cx, cy), size, (210, 80, 80), -1)

        cv2.circle(
            img,
            (cx, cy),
            size + 10,
            (255, 255, 255),
            2,
        )

    def draw_hud(self, img, phase_name, remaining):
        h, w = img.shape[:2]

        cv2.rectangle(
            img,
            (0, 0),
            (w, 58),
            (22, 22, 22),
            -1,
        )

        cv2.putText(
            img,
            f"FOCUS MISSION  |  {phase_name}",
            (18, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            img,
            f"Time remaining: {max(0, int(remaining))}s",
            (18, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (205, 205, 205),
            1,
        )

    # --------------------------------------------------------
    # Snapshot for Streamlit dashboard
    # --------------------------------------------------------
    def snapshot(self):
        with self.lock:
            return {
                "running": self.running,
                "protocol_start": self.protocol_start,
                "phase": self.current_phase,
                "trials": [asdict(t) for t in self.trials],
                "metrics": {
                    key: list(value)
                    if isinstance(value, list)
                    else value
                    for key, value in self.metrics.items()
                },
            }

    # --------------------------------------------------------
    # Main video callback
    # --------------------------------------------------------
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        now = time.perf_counter()

        with self.lock:
            start = self.protocol_start
            running = self.running

        if start is None or not running:
            cv2.putText(
                img,
                "READY",
                (
                    max(20, img.shape[1] // 2 - 50),
                    img.shape[0] // 2,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
            )

            return av.VideoFrame.from_ndarray(
                img,
                format="bgr24",
            )

        elapsed = now - start
        phase = self.get_phase(elapsed)

        if phase is None:
            self.stop_protocol()

            cv2.putText(
                img,
                "MISSION COMPLETE",
                (
                    max(20, img.shape[1] // 2 - 170),
                    img.shape[0] // 2,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (70, 220, 90),
                3,
            )

            return av.VideoFrame.from_ndarray(
                img,
                format="bgr24",
            )

        self.current_phase = phase["name"]

        # MediaPipe VIDEO timestamps must be strictly increasing.
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

        result = self.detector.detect_for_video(
            mp_image,
            timestamp_ms,
        )

        if result.pose_landmarks:
            self.process_pose(
                result.pose_landmarks[0],
                now,
            )

        # Trial handling.
        self.process_responses(now)
        self.expire_trial(now)

        with self.lock:
            trial = self.current_trial
            next_trial = self.next_trial_time

        # Start a new trial only after the previous one has ended.
        previous_finished = (
            trial is None
            or trial.response is not None
            or now - trial.onset > 1.50
        )

        if now >= next_trial and previous_finished:
            self.generate_trial(phase, now)

            with self.lock:
                trial = self.current_trial

        # Interface.
        self.draw_hud(
            img,
            phase["name"],
            phase["phase_remaining"],
        )

        # Subtle tracking points.
        if result.pose_landmarks:
            p = result.pose_landmarks[0]

            for idx in [0, 11, 12, 23, 24]:
                x = int(p[idx].x * img.shape[1])
                y = int(p[idx].y * img.shape[0])

                cv2.circle(
                    img,
                    (x, y),
                    3,
                    (0, 220, 220),
                    -1,
                )

        # Active stimulus.
        with self.lock:
            active_trial = self.current_trial

        if (
            active_trial
            and active_trial.response is None
            and now - active_trial.onset <= 1.50
        ):
            self.draw_stimulus(
                img,
                active_trial.stimulus,
            )

        # Position quality warning.
        if not self.position_valid:
            cv2.rectangle(
                img,
                (0, 0),
                (img.shape[1] - 1, img.shape[0] - 1),
                (0, 0, 255),
                8,
            )

            cv2.putText(
                img,
                "PLEASE SIT BACK AND FACE THE CAMERA",
                (20, img.shape[0] - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (0, 0, 255),
                2,
            )

        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24",
        )


# ============================================================
# RESULT CALCULATION
# ============================================================
def calculate_results(snapshot, participant_id):
    trials = snapshot["trials"]
    metrics = snapshot["metrics"]

    hits = sum(t["outcome"] == "hit" for t in trials)
    omissions = sum(t["outcome"] == "omission" for t in trials)
    commissions = sum(t["outcome"] == "commission" for t in trials)
    correct_rejections = sum(
        t["outcome"] == "correct_rejection"
        for t in trials
    )

    rts = [
        t["reaction_time"]
        for t in trials
        if t["reaction_time"] is not None
    ]

    valid_frames = metrics["valid_frames"]
    invalid_frames = metrics["invalid_frames"]

    total_frames = valid_frames + invalid_frames

    # These are descriptive measurements, NOT diagnostic cut-offs.
    results = {
        "participant_id": participant_id,
        "assessment_name": "Focus Mission Multimodal Attention Assessment",
        "assessment_duration_seconds": TOTAL_DURATION,
        "total_trials": len(trials),

        "performance": {
            "hits": hits,
            "omissions": omissions,
            "commissions": commissions,
            "correct_rejections": correct_rejections,
            "mean_reaction_time_seconds": mean_or_none(rts),
            "reaction_time_sd_seconds": std_or_none(rts),
            "response_count": len(rts),
        },

        "behavioral_activity": {
            "mean_velocity": mean_or_none(
                metrics["velocity"]
            ),
            "mean_acceleration": mean_or_none(
                metrics["acceleration"]
            ),
            "mean_movement_variance": mean_or_none(
                metrics["movement_variance"]
            ),
        },

        "camera_quality": {
            "valid_pose_frame_ratio": safe_ratio(
                valid_frames,
                total_frames,
            ),
            "valid_pose_frames": valid_frames,
            "invalid_pose_frames": invalid_frames,
        },

        "trials": trials,
    }

    return results


# ============================================================
# CSS
# ============================================================
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.3rem;
        max-width: 1250px;
    }

    .mission-card {
        padding: 22px;
        border-radius: 18px;
        border: 1px solid rgba(128,128,128,.25);
        background: rgba(128,128,128,.06);
        margin-bottom: 15px;
    }

    .response-note {
        text-align: center;
        padding: 14px;
        border-radius: 14px;
        background: rgba(70, 180, 90, .08);
        border: 1px solid rgba(70, 180, 90, .25);
    }

    .small-note {
        font-size: .88rem;
        opacity: .75;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================
st.title("🎯 Focus Mission")
st.caption(
    "Multimodal computerized attention and response-control assessment"
)

st.markdown(
    """
    <div class="mission-card">
    <b>What the participant does:</b>
    Watch the center of the screen and press the large response button
    whenever the target appears. The task becomes progressively more
    demanding while remaining literacy-neutral.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SETUP
# ============================================================
if not st.session_state["protocol_started"]:

    st.subheader("Participant Setup")

    participant_id = st.text_input(
        "Participant ID",
        value=st.session_state["participant_id"],
        placeholder="Example: P001",
    )

    st.markdown(
        """
        <div class="mission-card">
        <h4>Before starting</h4>
        <ul>
            <li>Sit comfortably in front of the camera.</li>
            <li>Keep your face and upper body visible.</li>
            <li>Use a quiet environment.</li>
            <li>Follow the visual target rule shown during training.</li>
            <li>Do not worry about speed; respond naturally.</li>
        </ul>
        <p class="small-note">
        The assessment is intended for research/screening support and is
        not a stand-alone medical diagnosis.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    agree = st.checkbox(
        "I understand the instructions and am ready to begin."
    )

    if st.button(
        "🚀 Start Focus Mission",
        type="primary",
        use_container_width=True,
    ):
        if not participant_id.strip():
            st.error("Please enter a participant ID.")
        elif not agree:
            st.error("Please confirm that you are ready.")
        else:
            st.session_state["participant_id"] = (
                participant_id.strip()
            )
            st.session_state["protocol_started"] = True
            st.session_state["protocol_finished"] = False
            st.session_state["protocol_start_wall"] = time.time()
            st.rerun()

    st.divider()

    st.subheader("📱 Optional Mobile Access")

    local_url = f"http://{get_local_ip()}:8501"

    try:
        import qrcode

        qr = qrcode.make(local_url)
        st.image(
            qr.get_image(),
            width=180,
            caption="Scan this QR code on the same Wi-Fi network",
        )
    except Exception:
        st.caption("Install qrcode if QR access is required.")


# ============================================================
# ASSESSMENT SCREEN
# ============================================================
if st.session_state["protocol_started"]:

    left, right = st.columns([3.3, 1])

    with left:
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
            key="focus-mission",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=rtc_config,
            video_processor_factory=MCPSProcessor,
            media_stream_constraints={
                "video": True,
                "audio": False,
            },
            async_processing=True,
        )

        if ctx and ctx.video_processor:
            processor = ctx.video_processor

            # Start the processor exactly once.
            if processor.protocol_start is None:
                processor.start_protocol()

            snapshot = processor.snapshot()

            if snapshot["protocol_start"] is not None:
                elapsed = (
                    time.perf_counter()
                    - snapshot["protocol_start"]
                )

                progress = min(
                    1.0,
                    elapsed / TOTAL_DURATION,
                )

                st.progress(progress)

                if elapsed >= TOTAL_DURATION:
                    processor.stop_protocol()
                    st.session_state["protocol_finished"] = True

    with right:
        st.markdown(
            """
            <div class="mission-card">
            <h3>🎯 Your Mission</h3>
            <p>Watch the center.</p>
            <p>When the target appears, press <b>RESPOND</b>.</p>
            <p>For other shapes, do nothing.</p>
            <p>Stay relaxed and respond naturally.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="response-note">Use the button below to respond.</div>',
            unsafe_allow_html=True,
        )

        if st.button(
            "🟢 RESPOND",
            type="primary",
            use_container_width=True,
        ):
            if ctx and ctx.video_processor:
                ctx.video_processor.register_response()

        st.caption(
            "The system records response timing automatically."
        )

        if ctx and ctx.video_processor:
            snap = ctx.video_processor.snapshot()

            total_trials = len(snap["trials"])
            completed_trials = sum(
                t["outcome"] != "pending"
                for t in snap["trials"]
            )

            st.metric(
                "Trials observed",
                total_trials,
            )

            st.metric(
                "Trials processed",
                completed_trials,
            )


# ============================================================
# CLINICIAN DASHBOARD
# ============================================================
st.divider()

with st.expander(
    "🔒 Clinician / Researcher Dashboard",
    expanded=False,
):
    st.warning(
        "This area is for authorized researchers/clinicians. "
        "Do not interpret a single composite score as an ADHD diagnosis."
    )

    if ctx and ctx.video_processor:
        snap = ctx.video_processor.snapshot()

        if not snap["running"] and snap["trials"]:
            results = calculate_results(
                snap,
                st.session_state["participant_id"],
            )

            st.session_state["export_payload"] = results

            st.subheader("Performance")

            p1, p2, p3, p4 = st.columns(4)

            perf = results["performance"]

            p1.metric(
                "Total Trials",
                results["total_trials"],
            )

            p2.metric(
                "Hits",
                perf["hits"],
            )

            p3.metric(
                "Omissions",
                perf["omissions"],
            )

            p4.metric(
                "Commissions",
                perf["commissions"],
            )

            st.subheader("Reaction Time")

            r1, r2 = st.columns(2)

            r1.metric(
                "Mean RT",
                (
                    f'{perf["mean_reaction_time_seconds"]:.3f} s'
                    if perf["mean_reaction_time_seconds"] is not None
                    else "—"
                ),
            )

            r2.metric(
                "RT Variability",
                (
                    f'{perf["reaction_time_sd_seconds"]:.3f} s'
                    if perf["reaction_time_sd_seconds"] is not None
                    else "—"
                ),
            )

            st.subheader("Behavioral Activity")

            behavior = results["behavioral_activity"]

            b1, b2, b3 = st.columns(3)

            b1.metric(
                "Mean Velocity",
                (
                    f'{behavior["mean_velocity"]:.4f}'
                    if behavior["mean_velocity"] is not None
                    else "—"
                ),
            )

            b2.metric(
                "Mean Acceleration",
                (
                    f'{behavior["mean_acceleration"]:.4f}'
                    if behavior["mean_acceleration"] is not None
                    else "—"
                ),
            )

            b3.metric(
                "Movement Variance",
                (
                    f'{behavior["mean_movement_variance"]:.4f}'
                    if behavior["mean_movement_variance"] is not None
                    else "—"
                ),
            )

            st.subheader("Camera Quality")

            quality = results["camera_quality"]

            st.metric(
                "Valid Pose Frame Ratio",
                f'{quality["valid_pose_frame_ratio"] * 100:.1f}%',
            )

            st.subheader("Raw Trial Data")

            st.dataframe(
                results["trials"],
                use_container_width=True,
            )

            json_data = json.dumps(
                results,
                indent=2,
            )

            st.download_button(
                "⬇️ Export Complete JSON",
                data=json_data,
                file_name=(
                    f'{results["participant_id"]}'
                    '_focus_mission_results.json'
                ),
                mime="application/json",
                use_container_width=True,
            )

            st.caption(
                "ADFI coefficient optimization should be performed "
                "after labelled pilot data are collected; this interface "
                "does not fabricate or apply diagnostic thresholds."
            )

        else:
            st.info(
                "Complete the assessment before reviewing results."
            )
