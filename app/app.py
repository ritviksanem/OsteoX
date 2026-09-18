# ============================================================
# KneeSense NER - SIH26004
# AI-Assisted Early Screening for Osteoarthritis Risk Markers
# ============================================================

import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import av
import time
import math
import threading
import joblib
import urllib.request

from pathlib import Path
from scipy.signal import find_peaks
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import sys
from pathlib import Path

# Add project root to path if needed so scr is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
  sys.path.append(str(PROJECT_ROOT))

from scr.db import (
    get_all_patients,
    get_patient_history,
    init_db,
    log_screening_visit,
    register_patient,
)

# Initialize tables on app launch
init_db()

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="KneeSense NER",
    page_icon="🦵",
    layout="wide"
)

#Ensure Session State Tracking
if "current_patient_id" not in st.session_state:
  st.session_state["current_patient_id"] = None
if "current_patient_name" not in st.session_state:
  st.session_state["current_patient_name"] = "Unknown"
if "visit_logged" not in st.session_state:
  st.session_state["visit_logged"] = False

# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

DATA_FOLDER = ROOT / "data"
FEATURE_FOLDER = DATA_FOLDER / "features"
PROCESSED_FOLDER = DATA_FOLDER / "processed"
MODEL_FOLDER = ROOT / "models"

FEATURE_FOLDER.mkdir(parents=True, exist_ok=True)
PROCESSED_FOLDER.mkdir(parents=True, exist_ok=True)
MODEL_FOLDER.mkdir(parents=True, exist_ok=True)

QUESTIONNAIRE_FILE = FEATURE_FOLDER / "questionnaire_features.csv"
WALKING_FEATURE_FILE = PROCESSED_FOLDER / "walking_features.csv"
#SCREENING_FILE = PROCESSED_FOLDER / "screening_records.csv"
MODEL_FILE = MODEL_FOLDER / "oa_risk_model.pkl"

from scr.db import get_all_patients, get_patient_history, init_db, log_screening_visit, register_patient

init_db()

# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    "patient_name": "",
    "questionnaire_saved": False,
    "test_finished": False,
    "movement_result": None,
    "movement_saved": False,
    "screening_done": False,
    "screening_result": None,
    "test_id": 0,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# HEADER
# ============================================================

st.title("🦵 KneeSense NER")
st.subheader("AI-Assisted Early Screening for Osteoarthritis Risk Markers")
st.caption("Portable • Offline • AI-assisted • Preliminary screening")

st.markdown(
    """
KneeSense NER combines **patient symptoms + camera-based pose analysis +
movement symmetry** to produce a **LOW / MODERATE / HIGH preliminary risk
category**.

> ⚠️ **Research screening prototype**: this system does not provide a definitive diagnosis. 
>The ML risk model is trained on 8,260 real patient records from the NIH Osteoarthritis Initiative (OAI) dataset for preliminary clinical triage
"""
)

st.divider()


# ============================================================
# KNEE ANGLE CALCULATION
# ============================================================

def calculate_angle(a, b, c):
    angle = math.degrees(
        math.atan2(c[1] - b[1], c[0] - b[0])
        - math.atan2(a[1] - b[1], a[0] - b[0])
    )

    angle = abs(angle)

    if angle > 180:
        angle = 360 - angle

    return angle


# ============================================================
# MOVEMENT FEATURE CALCULATION
# ============================================================

def calculate_features(left_angles, right_angles, timestamps):

    if len(left_angles) < 20 or len(right_angles) < 20:
        return {
            "error": True,
            "message": (
                f"Not enough valid pose data. "
                f"Left samples: {len(left_angles)}, "
                f"Right samples: {len(right_angles)}"
            )
        }

    left = np.asarray(left_angles, dtype=float)
    right = np.asarray(right_angles, dtype=float)
    times = np.asarray(timestamps, dtype=float)

    # --------------------------------------------------------
    # Median smoothing
    # --------------------------------------------------------

    def smooth(signal, window=5):

        if len(signal) < window:
            return signal.copy()

        result = signal.copy()
        half = window // 2

        for i in range(half, len(signal) - half):
            result[i] = np.median(
                signal[i - half:i + half + 1]
            )

        return result

    left_s = smooth(left)
    right_s = smooth(right)

    # --------------------------------------------------------
    # Basic movement statistics
    # --------------------------------------------------------

    left_max = float(np.max(left_s))
    left_min = float(np.min(left_s))
    left_rom = float(left_max - left_min)
    left_avg = float(np.mean(left_s))

    right_max = float(np.max(right_s))
    right_min = float(np.min(right_s))
    right_rom = float(right_max - right_min)
    right_avg = float(np.mean(right_s))

    # --------------------------------------------------------
    # ROM asymmetry
    # --------------------------------------------------------

    max_rom = max(left_rom, right_rom, 1e-6)

    asymmetry = (
        abs(left_rom - right_rom)
        / max_rom
        * 100.0
    )

    # --------------------------------------------------------
    # Duration / sample rate
    # --------------------------------------------------------

    duration = float(times[-1] - times[0])

    if duration <= 0:
        duration = 1.0

    sample_rate = len(times) / duration

    # --------------------------------------------------------
    # Gait event detection
    # --------------------------------------------------------

    def gait_events(signal):

        distance = max(
            3,
            int(sample_rate * 0.6)
        )

        valleys, _ = find_peaks(
            -signal,
            distance=distance,
            prominence=2.5
        )

        valid = [
            int(i)
            for i in valleys
            if 110 <= signal[i] <= 178
        ]

        cycles = []

        for i in range(1, len(valid)):

            dt = float(
                times[valid[i]]
                - times[valid[i - 1]]
            )

            if 0.6 <= dt <= 2.5:
                cycles.append(dt)

        return valid, cycles

    left_events, left_cycles = gait_events(left_s)
    right_events, right_cycles = gait_events(right_s)

    # --------------------------------------------------------
    # Consistency
    # --------------------------------------------------------

    def consistency(cycles):

        if len(cycles) < 2:
            return 0.0

        arr = np.asarray(cycles, dtype=float)

        mean_cycle = float(np.mean(arr))

        if mean_cycle <= 0:
            return 0.0

        cv = float(
            np.std(arr) / mean_cycle
        )

        return float(
            np.clip(100 - cv * 100, 0, 100)
        )

    left_consistency = consistency(left_cycles)
    right_consistency = consistency(right_cycles)

    # --------------------------------------------------------
    # Cycle time
    # --------------------------------------------------------

    left_cycle = (
        float(np.mean(left_cycles))
        if left_cycles
        else 0.0
    )

    right_cycle = (
        float(np.mean(right_cycles))
        if right_cycles
        else 0.0
    )

    # --------------------------------------------------------
    # Cadence
    #
    # Informational only.
    # NOT used by ML model.
    # --------------------------------------------------------

    left_cadence = (
        len(left_cycles) * 2
        / duration
        * 60
        if duration > 0
        else 0.0
    )

    right_cadence = (
        len(right_cycles) * 2
        / duration
        * 60
        if duration > 0
        else 0.0
    )

    # --------------------------------------------------------
    # Data quality
    # --------------------------------------------------------

    valid_samples = min(
        len(left),
        len(right)
    )

    if (
        valid_samples >= 300
        and sample_rate >= 10
    ):
        quality = "GOOD"

    elif (
        valid_samples >= 150
        and sample_rate >= 7
    ):
        quality = "FAIR"

    else:
        quality = "POOR"

    # --------------------------------------------------------
    # Return
    # --------------------------------------------------------

    return {
        "error": False,

        "left_max_angle": left_max,
        "left_min_angle": left_min,
        "left_rom": left_rom,
        "left_average_angle": left_avg,

        "right_max_angle": right_max,
        "right_min_angle": right_min,
        "right_rom": right_rom,
        "right_average_angle": right_avg,

        "rom_asymmetry": float(asymmetry),

        "left_gait_events": len(left_events),
        "right_gait_events": len(right_events),

        "left_average_cycle_time": left_cycle,
        "right_average_cycle_time": right_cycle,

        "left_cadence": float(left_cadence),
        "right_cadence": float(right_cadence),

        "left_consistency": left_consistency,
        "right_consistency": right_consistency,

        "valid_samples": valid_samples,
        "sample_rate": float(sample_rate),

        "quality": quality,
    }


# ============================================================
# WEBRTC VIDEO PROCESSOR
# ============================================================

class VideoProcessor:

    def __init__(self):
        self.start_time = None
        self.finished = False
        self.finishing = False
        self.result = None
        self.left_angles = []
        self.right_angles = []
        self.timestamps = []
        self.lock = threading.Lock()
        self.last_ts = -1

        # Automatically download the modern pose model on first run
        model_path = MODEL_FOLDER / "pose_landmarker_lite.task"
        if not model_path.exists():
            print("Downloading MediaPipe Pose Landmarker model...")
            urllib.request.urlretrieve(
                "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
                str(model_path)
            )

        # Initialize the modern Tasks API
        base_options = python.BaseOptions(model_asset_path=str(model_path))
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")

        # ----------------------------------------------------
        # Start timer when actual camera frames arrive.
        # ----------------------------------------------------
        if self.start_time is None:
            self.start_time = time.time()

        elapsed = time.time() - self.start_time

        # ----------------------------------------------------
        # Finished state
        # ----------------------------------------------------
        with self.lock:
            finished = self.finished
            finishing = self.finishing

        if finished:
            cv2.putText(
                image, "TEST COMPLETE", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3
            )
            return av.VideoFrame.from_ndarray(image, format="bgr24")

        if finishing:
            cv2.putText(
                image, "FINALIZING TEST...", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 3
            )
            return av.VideoFrame.from_ndarray(image, format="bgr24")

        # ----------------------------------------------------
        # Modern MediaPipe Processing
        # ----------------------------------------------------
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        # Calculate monotonically increasing timestamp in milliseconds
        timestamp_ms = int(elapsed * 1000)
        if timestamp_ms <= self.last_ts:
            timestamp_ms = self.last_ts + 1
        self.last_ts = timestamp_ms

        results = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        if results.pose_landmarks and len(results.pose_landmarks) > 0:
            lm = results.pose_landmarks[0] 

            # Map the coordinate array (23-28 are hips to ankles)
            lh, lk, la = lm[23], lm[25], lm[27]
            rh, rk, ra = lm[24], lm[26], lm[28]

            visibility = [
                lh.visibility, lk.visibility, la.visibility, 
                rh.visibility, rk.visibility, ra.visibility
            ]

            # ------------------------------------------------
            # Only store reliable pose measurements.
            # ------------------------------------------------
            if min(visibility) >= 0.45:
                left_angle = calculate_angle((lh.x, lh.y), (lk.x, lk.y), (la.x, la.y))
                right_angle = calculate_angle((rh.x, rh.y), (rk.x, rk.y), (ra.x, ra.y))

                with self.lock:
                    self.left_angles.append(float(left_angle))
                    self.right_angles.append(float(right_angle))
                    self.timestamps.append(float(elapsed))

                # Custom reliable full-body wireframe drawing
                h, w, _ = image.shape

                # MediaPipe Pose Topology (Torso, Arms, Legs & Feet)
                POSE_CONNECTIONS = [
                    # Torso
                    (11, 12), (11, 23), (12, 24), (23, 24),
                    # Left Arm
                    (11, 13), (13, 15),
                    # Right Arm
                    (12, 14), (14, 16),
                    # Left Leg & Foot
                    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
                    # Right Leg & Foot
                    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
                ]

                # Map visible landmarks to pixel coordinates using 'lm'
                pts = {}
                for idx, pt in enumerate(lm):
                    vis = getattr(pt, "visibility", 1.0)
                    if vis > 0.4:
                        pts[idx] = (int(pt.x * w), int(pt.y * h))

                # 1. Draw full body skeleton lines (Sleek clinic silver)
                for start_idx, end_idx in POSE_CONNECTIONS:
                    if start_idx in pts and end_idx in pts:
                        cv2.line(image, pts[start_idx], pts[end_idx], (220, 220, 220), 2)

                # 2. Draw standard joint keypoints (Small red markers)
                for idx, pt in pts.items():
                    if idx not in [25, 26]:  # Exclude knees from generic dots
                        cv2.circle(image, pt, 4, (0, 0, 255), cv2.FILLED)

                # 3. Highlight Knee Joints prominently (Vivid Green + Outer Ring)
                for knee_idx in [25, 26]:
                    if knee_idx in pts:
                        cv2.circle(image, pts[knee_idx], 8, (0, 255, 0), cv2.FILLED)
                        cv2.circle(image, pts[knee_idx], 11, (255, 255, 255), 2)

                # 4. Telemetry Overlay
                cv2.putText(
                    image, f"Left Knee: {left_angle:.1f} deg", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                )
                cv2.putText(
                    image, f"Right Knee: {right_angle:.1f} deg", (20, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                )

        # ----------------------------------------------------
        # Samples
        # ----------------------------------------------------
        with self.lock:
            sample_count = min(len(self.left_angles), len(self.right_angles))

        cv2.putText(
            image, f"Valid Samples: {sample_count}", (20, 155),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2
        )

        # ----------------------------------------------------
        # Countdown
        # ----------------------------------------------------
        remaining = max(0, int(30 - elapsed))

        cv2.putText(
            image, f"Movement Test: {remaining}s", (20, 195),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2
        )

        cv2.putText(
            image, "Walk naturally - keep full body visible", (20, 230),
            cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2
        )

        # ----------------------------------------------------
        # Finish exactly once after 30 seconds.
        # ----------------------------------------------------
        if elapsed >= 30:
            with self.lock:
                if not self.finishing and not self.finished:
                    self.finishing = True
                    left_copy = list(self.left_angles)
                    right_copy = list(self.right_angles)
                    time_copy = list(self.timestamps)
                else:
                    left_copy = None
                    right_copy = None
                    time_copy = None

            if left_copy is not None:
                try:
                    result = calculate_features(left_copy, right_copy, time_copy)
                except Exception as exc:
                    result = {
                        "error": True,
                        "message": f"Feature calculation failed: {exc}"
                    }

                with self.lock:
                    self.result = result
                    self.finished = True
                    self.finishing = False

                try:
                    self.landmarker.close()
                except Exception:
                    pass

                cv2.putText(
                    image, "TEST COMPLETE", (20, 275),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 3
                )

        return av.VideoFrame.from_ndarray(
            image,
            format="bgr24"
        )
# ============================================================
# MOVEMENT REPORT
# ============================================================

def show_movement_report(result):

    st.divider()

    st.header("3️⃣ Movement Analysis")

    st.success(
        "✅ 30-second movement assessment completed."
    )

    st.markdown(
        f"**Patient:** {st.session_state.patient_name}"
    )

    st.write(
        "Measurements extracted from camera-based pose analysis:"
    )

    # --------------------------------------------------------
    # Knees
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### 🦵 Left Knee")

        st.metric(
            "Maximum Angle",
            f"{result['left_max_angle']:.2f}°"
        )

        st.metric(
            "Minimum Angle",
            f"{result['left_min_angle']:.2f}°"
        )

        st.metric(
            "Range of Motion",
            f"{result['left_rom']:.2f}°"
        )

        st.metric(
            "Average Angle",
            f"{result['left_average_angle']:.2f}°"
        )

        st.metric(
            "Consistency",
            f"{result['left_consistency']:.2f}%"
        )

    with col2:

        st.markdown("### 🦵 Right Knee")

        st.metric(
            "Maximum Angle",
            f"{result['right_max_angle']:.2f}°"
        )

        st.metric(
            "Minimum Angle",
            f"{result['right_min_angle']:.2f}°"
        )

        st.metric(
            "Range of Motion",
            f"{result['right_rom']:.2f}°"
        )

        st.metric(
            "Average Angle",
            f"{result['right_average_angle']:.2f}°"
        )

        st.metric(
            "Consistency",
            f"{result['right_consistency']:.2f}%"
        )

    st.divider()

    # --------------------------------------------------------
    # Symmetry
    # --------------------------------------------------------

    st.markdown("### ⚖️ Movement Symmetry")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "ROM Asymmetry",
            f"{result['rom_asymmetry']:.2f}%"
        )

    with col2:
        st.metric(
            "Left Gait Events",
            result["left_gait_events"]
        )

    with col3:
        st.metric(
            "Right Gait Events",
            result["right_gait_events"]
        )

    # --------------------------------------------------------
    # Gait
    # --------------------------------------------------------

    st.markdown("### 🚶 Gait Information")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Left Cycle",
            f"{result['left_average_cycle_time']:.2f}s"
        )

    with col2:
        st.metric(
            "Right Cycle",
            f"{result['right_average_cycle_time']:.2f}s"
        )

    with col3:
        st.metric(
            "Left Cadence",
            f"{result['left_cadence']:.1f}"
        )

    with col4:
        st.metric(
            "Right Cadence",
            f"{result['right_cadence']:.1f}"
        )

    st.caption(
        "Cadence is informational only and is NOT used by the ML model."
    )

    # --------------------------------------------------------
    # Data quality
    # --------------------------------------------------------

    st.markdown("### 🧪 Data Quality")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Valid Samples",
            result["valid_samples"]
        )

    with col2:
        st.metric(
            "Sample Rate",
            f"{result['sample_rate']:.2f} FPS"
        )

    with col3:

        quality = result["quality"]

        if quality == "GOOD":
            st.success(f"Quality: {quality}")

        elif quality == "FAIR":
            st.warning(f"Quality: {quality}")

        else:
            st.error(f"Quality: {quality}")

    # --------------------------------------------------------
    # Save movement features
    # --------------------------------------------------------

    movement_columns = [
        "left_max_angle",
        "left_min_angle",
        "left_rom",
        "left_average_angle",

        "right_max_angle",
        "right_min_angle",
        "right_rom",
        "right_average_angle",

        "rom_asymmetry",

        "left_gait_events",
        "right_gait_events",

        "left_average_cycle_time",
        "right_average_cycle_time",

        "left_cadence",
        "right_cadence",

        "left_consistency",
        "right_consistency",

        "valid_samples",
        "sample_rate",
    ]

    if not st.session_state.movement_saved:

        movement_row = {
            column: result.get(column, 0)
            for column in movement_columns
        }

        pd.DataFrame(
            [movement_row]
        ).to_csv(
            WALKING_FEATURE_FILE,
            index=False
        )

        st.session_state.movement_saved = True

        st.success(
            "✅ Movement features saved successfully."
        )


# ============================================================
# 1. PATIENT QUESTIONNAIRE
# ============================================================

st.header("1️⃣ Patient Questionnaire")

st.write(
    "Enter the patient's basic information and symptoms "
    "before starting the movement assessment."
)

# Patient mode selection (outside form so dropdown triggers an immediate rerun)
existing_patients = get_all_patients()
patient_mode = st.radio(
    "Patient Registration Type:",
    ["New Patient", "Existing Patient"],
    horizontal=True,
)

selected_pid = None
default_name = ""
default_age = 45
default_sex_idx = 0

if patient_mode == "Existing Patient":
  if existing_patients:
    patient_dict = {
        f"{p[0]} - {p[1]} (Age: {p[2]})": p for p in existing_patients
    }
    selected_label = st.selectbox(
        "Select Existing Patient:", list(patient_dict.keys())
    )
    selected_record = patient_dict[selected_label]
    selected_pid = selected_record[0]
    default_name = selected_record[1]
    default_age = int(selected_record[2])
    default_sex_idx = 1 if selected_record[3] == 1 else 0
    st.info(f"Selected Patient ID: `{selected_pid}`")
  else:
    st.warning(
        "No existing patients found in database. Please register as New"
        " Patient."
    )
    patient_mode = "New Patient"

with st.form("questionnaire_form"):

  patient_name = st.text_input(
      "Patient Name",
      value=default_name,
      placeholder="Enter patient name",
      disabled=(patient_mode == "Existing Patient"),
  )

  col1, col2 = st.columns(2)

  with col1:
    age = st.number_input(
        "Age", min_value=18, max_value=100, value=default_age
    )

    sex = st.selectbox(
        "Sex", ["Male", "Female", "Other"], index=default_sex_idx
    )

    pain = st.slider("Current knee pain (0–10)", 0, 10, 0)

    pain_duration = st.selectbox(
        "Pain duration", ["None", "<3 months", "3–12 months", ">1 year"]
    )

    morning_stiffness = st.selectbox("Morning stiffness?", ["No", "Yes"])

  with col2:
    stairs_difficulty = st.selectbox(
        "Difficulty climbing stairs", ["None", "Mild", "Severe"]
    )

    walking_difficulty = st.selectbox(
        "Difficulty walking", ["None", "Mild", "Severe"]
    )

    previous_injury = st.selectbox("Previous knee injury?", ["No", "Yes"])

    previous_surgery = st.selectbox("Previous knee surgery?", ["No", "Yes"])

    family_history = st.selectbox(
        "Family history of osteoarthritis?", ["No", "Yes"]
    )

  submitted = st.form_submit_button(
      "💾 Save Patient & Questionnaire", use_container_width=True
  )


# ============================================================
# SAVE QUESTIONNAIRE
# ============================================================

if submitted:

  if not patient_name.strip():
    st.error("❌ Please enter the patient name.")
  else:
    sex_num = {"Male": 0, "Female": 1, "Other": 2}[sex]

    # Assign or look up permanent Patient ID
    if patient_mode == "New Patient" or not selected_pid:
      patient_id = register_patient(
          name=patient_name.strip(), age=int(age), sex=sex_num
      )
    else:
      patient_id = selected_pid

    questionnaire_data = {
        "patient_id": patient_id,
        "patient_name": patient_name.strip(),
        "age": int(age),
        "sex": sex_num,
        "pain": int(pain),
        "pain_duration": {
            "None": 0,
            "<3 months": 1,
            "3–12 months": 2,
            ">1 year": 3,
        }[pain_duration],
        "morning_stiffness": 1 if morning_stiffness == "Yes" else 0,
        "stairs_difficulty": {"None": 0, "Mild": 1, "Severe": 2}[
            stairs_difficulty
        ],
        "walking_difficulty": {"None": 0, "Mild": 1, "Severe": 2}[
            walking_difficulty
        ],
        "previous_injury": 1 if previous_injury == "Yes" else 0,
        "previous_surgery": 1 if previous_surgery == "Yes" else 0,
        "family_history": 1 if family_history == "Yes" else 0,
    }

    # Save active session buffer for downstream steps
    pd.DataFrame([questionnaire_data]).to_csv(QUESTIONNAIRE_FILE, index=False)

    # Persist ID and Name across session states
    st.session_state.patient_id = patient_id
    st.session_state.patient_name = patient_name.strip()
    st.session_state.questionnaire_saved = True
    st.session_state.visit_logged = False

    # Clear old movement/AI results
    st.session_state.test_finished = False
    st.session_state.movement_result = None
    st.session_state.movement_saved = False
    st.session_state.screening_done = False
    st.session_state.screening_result = None

    # New WebRTC key
    st.session_state.test_id += 1

    st.success(
        f"✅ Patient '{patient_name.strip()}' (ID: `{patient_id}`) and"
        " questionnaire saved successfully."
    )


if st.session_state.get("questionnaire_saved", False):
  current_id = st.session_state.get("patient_id", "N/A")
  st.success(
      f"👤 Current Patient: **{st.session_state.patient_name}** | ID:"
      f" `{current_id}`"
  )

st.divider()

# ============================================================
# 2. MOVEMENT ASSESSMENT
# ============================================================

st.header("2️⃣ Movement Assessment")

if not st.session_state.questionnaire_saved:

    st.warning(
        "🔒 Complete and save the patient questionnaire first."
    )

else:

    st.write(
        "Stand approximately 2–3 meters away from the camera "
        "so your full body is visible."
    )

    st.info(
        "🚶 The test runs automatically for 30 seconds. "
        "Walk naturally while keeping both legs visible."
    )

    camera_key = (
        f"kneesense-camera-{st.session_state.test_id}"
    )

    # --------------------------------------------------------
    # The camera and monitoring are inside a Streamlit
    # fragment so the main script is never blocked.
    # --------------------------------------------------------

    @st.fragment(run_every="500ms")
    def movement_camera():

        # If the test has already finished, do not recreate
        # the WebRTC component.
        if st.session_state.test_finished:
            return

        ctx = webrtc_streamer(
            key=camera_key,
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            ),
            video_processor_factory=VideoProcessor,
            media_stream_constraints={
                "video": True,
                "audio": False
            },
            async_processing=True
        )

        processor = ctx.video_processor

        # ----------------------------------------------------
        # Camera not started yet
        # ----------------------------------------------------

        if processor is None:

            st.info(
                "📷 Camera is ready. Click START in the "
                "camera window and allow browser permission."
            )

            return

        # ----------------------------------------------------
        # Read processor state safely.
        # ----------------------------------------------------

        with processor.lock:

            finished = processor.finished
            result = processor.result
            start_time = processor.start_time

            samples = min(
                len(processor.left_angles),
                len(processor.right_angles)
            )

        # ----------------------------------------------------
        # Camera connected but not started.
        # ----------------------------------------------------

        if start_time is None:

            st.info(
                "📷 Camera connected. "
                "Click START to begin the 30-second test."
            )

            return

        # ----------------------------------------------------
        # Test still running.
        # ----------------------------------------------------

        if not finished:

            elapsed = (
                time.time()
                - start_time
            )

            remaining = max(
                0,
                int(30 - elapsed)
            )

            st.info(
                f"🚶 Test running — "
                f"**{remaining} seconds remaining**"
            )

            st.caption(
                f"Valid pose samples detected: {samples}"
            )

            return

        # ----------------------------------------------------
        # TEST FINISHED
        # ----------------------------------------------------

        if result is None:

            st.error(
                "❌ Test finished but no result was generated."
            )

            return

        # Store result in session state FIRST.
        # The important part is the full-app rerun below:
        # fragment reruns do not automatically refresh widgets
        # that are outside the fragment. Without this, the AI
        # section can remain stuck waiting for the movement test.
        st.session_state.movement_result = result
        st.session_state.test_finished = True

        # Stop the WebRTC stream.
        try:
            ctx.stop()
        except Exception:
            pass

        # Force the WHOLE Streamlit app to rerun.
        # This makes the movement report and AI screening section
        # appear immediately after the 30-second test.
        st.rerun(scope="app")


    movement_camera()


# ============================================================
# SHOW REPORT AGAIN AFTER NORMAL APP RERUN
# ============================================================

# When the AI button or another Streamlit action causes a full
# rerun, preserve the movement report.
if (
    st.session_state.test_finished
    and st.session_state.movement_result is not None
):

    # The fragment already renders it during completion.
    # On later full reruns it must be rendered here as well.
    show_movement_report(
        st.session_state.movement_result
    )


# ============================================================
# RESTART MOVEMENT TEST
# ============================================================

if (
    st.session_state.questionnaire_saved
    and st.session_state.test_finished
):

    if st.button(
        "🔄 Restart Movement Test",
        use_container_width=True
    ):

        st.session_state.test_id += 1

        st.session_state.test_finished = False
        st.session_state.movement_result = None
        st.session_state.movement_saved = False

        st.session_state.screening_done = False
        st.session_state.screening_result = None

        st.rerun()


st.divider()


# ============================================================
# 4. AI RISK SCREENING
# ============================================================

st.header("4️⃣ AI Risk Screening")

st.write(
    "The AI combines questionnaire information and "
    "camera-based movement features to generate a "
    "preliminary risk category."
)

if not st.session_state.questionnaire_saved:
    st.warning("🔒 Complete the questionnaire first.")

elif not st.session_state.test_finished:
    st.warning("🔒 Complete the 30-second movement assessment first.")

elif not MODEL_FILE.exists():
    st.error(
        "❌ AI model not found.\n\n"
        f"Expected location:\n{MODEL_FILE}"
    )

elif st.session_state.screening_result is None:
    # ------------------------------------------------------------
    # Run the AI automatically once movement analysis is complete.
    # This removes the extra button and guarantees that the result
    # is shown in Step 4 after the camera test finishes.
    # ------------------------------------------------------------
    with st.spinner("🤖 Running AI risk screening..."):
        try:
            movement_df = pd.read_csv(WALKING_FEATURE_FILE)
            questionnaire_df = pd.read_csv(QUESTIONNAIRE_FILE)

            if movement_df.empty:
                raise ValueError("Movement feature file is empty.")
            if questionnaire_df.empty:
                raise ValueError("Questionnaire file is empty.")

            movement = movement_df.iloc[-1]
            questionnaire = questionnaire_df.iloc[-1]

            feature_names = [
                "left_rom",
                "right_rom",
                "rom_asymmetry",
                "left_average_angle",
                "right_average_angle",
                "left_consistency",
                "right_consistency",
                "age",
                "sex",
                "pain",
                "pain_duration",
                "morning_stiffness",
                "stairs_difficulty",
                "walking_difficulty",
                "previous_injury",
                "previous_surgery",
                "family_history",
            ]

            model_input = pd.DataFrame([[
                movement["left_rom"],
                movement["right_rom"],
                movement["rom_asymmetry"],
                movement["left_average_angle"],
                movement["right_average_angle"],
                movement["left_consistency"],
                movement["right_consistency"],
                questionnaire["age"],
                questionnaire["sex"],
                questionnaire["pain"],
                questionnaire["pain_duration"],
                questionnaire["morning_stiffness"],
                questionnaire["stairs_difficulty"],
                questionnaire["walking_difficulty"],
                questionnaire["previous_injury"],
                questionnaire["previous_surgery"],
                questionnaire["family_history"],
            ]], columns=feature_names)

            model = joblib.load(MODEL_FILE)

            # Align columns with the model when feature names are available.
            if hasattr(model, "feature_names_in_"):
                expected_features = list(model.feature_names_in_)
                missing = [x for x in expected_features if x not in model_input.columns]
                if missing:
                    raise ValueError(
                        "The trained model expects missing features: "
                        + ", ".join(missing)
                    )
                model_input = model_input[expected_features]

            prediction = model.predict(model_input)[0]

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(model_input)[0]
                classes = list(model.classes_)
                probability_dict = {
                    str(label).upper(): float(probability)
                    for label, probability in zip(classes, probabilities)
                }
            else:
                probability_dict = {str(prediction).upper(): 1.0}

            risk = (
                str(prediction)
                .upper()
                .replace("RISK", "")
                .strip()
            )

            if risk not in {"LOW", "MODERATE", "HIGH"}:
                raise ValueError(f"Unexpected model output: {prediction}")

            recommendations = {
                "LOW": (
                    "Low preliminary risk markers. Maintain healthy activity "
                    "and monitor symptoms. Seek clinical evaluation if persistent "
                    "pain or functional difficulty develops."
                ),
                "MODERATE": (
                    "Moderate preliminary risk markers. Consider evaluation by a "
                    "qualified healthcare professional, especially if pain, stiffness, "
                    "or functional difficulty persists."
                ),
                "HIGH": (
                    "High preliminary risk markers. Clinical evaluation is recommended "
                    "for further assessment."
                ),
            }

            screening = {
                "risk": risk,
                "probabilities": probability_dict,
                "recommendation": recommendations[risk],
            }

            st.session_state.screening_result = screening
            st.session_state.screening_done = True

            # Save screening visit to SQLite
            patient_id = st.session_state.get("patient_id")
            features_dict = (
                model_input.iloc[0].to_dict()
                if isinstance(model_input, pd.DataFrame)
                else dict(model_input)
            )

            # Fallback registration if session state lost ID
            if not patient_id:
              patient_id = register_patient(
                  name=st.session_state.get("patient_name", "Unknown"),
                  age=int(features_dict.get("age", 45)),
                  sex=int(features_dict.get("sex", 0)),
              )
              st.session_state.patient_id = patient_id

            if not st.session_state.get("visit_logged", False):
              log_screening_visit(
                  patient_id=patient_id,
                  kinematics=features_dict,
                  questionnaire=features_dict,
                  risk=risk,
                  probabilities=probability_dict,
                  notes="Automated clinical screening triage",
              )
              st.session_state.visit_logged = True
            # Do not call st.rerun() here. The current run continues directly
            # into Step 5, so the result appears immediately.

        except Exception as exc:
            st.error("❌ Error during AI screening.")
            st.exception(exc)

# If a result already exists, Step 4 confirms it is ready.
if st.session_state.screening_result is not None:
    st.success("✅ AI screening completed successfully. See Step 5 below for the result.")


# ============================================================
# 5. FINAL SCREENING RESULT
# ============================================================

if (
    st.session_state.screening_done
    and st.session_state.screening_result is not None
):

    screening = (
        st.session_state.screening_result
    )

    risk = screening["risk"]

    st.divider()

    st.header(
        "5️⃣ Final Screening Result"
    )

    st.markdown(
        f"### 👤 Patient: "
        f"**{st.session_state.patient_name}**"
    )

    # --------------------------------------------------------
    # Risk appearance
    # --------------------------------------------------------

    if risk == "LOW":

        emoji = "🟢"
        title = "LOW RISK"
        message = (
            "Low preliminary risk markers detected."
        )

        background = "#d4edda"
        border = "#28a745"
        text = "#155724"

    elif risk == "MODERATE":

        emoji = "🟡"
        title = "MODERATE RISK"
        message = (
            "Moderate preliminary risk markers detected."
        )

        background = "#fff3cd"
        border = "#ffc107"
        text = "#856404"

    else:

        emoji = "🔴"
        title = "HIGH RISK"
        message = (
            "High preliminary risk markers detected."
        )

        background = "#f8d7da"
        border = "#dc3545"
        text = "#721c24"

    # --------------------------------------------------------
    # Large result card
    # --------------------------------------------------------
    # Native Streamlit components are used instead of raw HTML.
    # This prevents the result from being displayed as a code block.

    with st.container(border=True):
        _, center, _ = st.columns([1, 2, 1])

        with center:
            st.markdown(f"# {emoji}")
            st.markdown(f"## {title}")

            if risk == "LOW":
                st.success(message)
            elif risk == "MODERATE":
                st.warning(message)
            else:
                st.error(message)

    # Probability
    # --------------------------------------------------------

    st.markdown(
        "### 📊 AI Model Probability"
    )

    probability_dict = (
        screening["probabilities"]
    )

    probability_table = pd.DataFrame({

        "Risk Category": [
            str(label).upper()
            for label
            in probability_dict.keys()
        ],

        "Probability (%)": [
            round(
                float(value) * 100,
                2
            )
            for value
            in probability_dict.values()
        ]
    })

    st.dataframe(
        probability_table,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # Recommendation
    # --------------------------------------------------------

    st.markdown(
        "### 📋 Recommendation"
    )

    st.info(
        screening["recommendation"]
    )

    st.warning(
        "⚠️ This is a preliminary screening result "
        "and NOT a diagnosis."
    )

# ============================================================
# LONGITUDINAL PATIENT HISTORY
# ============================================================
st.markdown("---")
st.subheader("📋 Longitudinal Visit History")

current_pid = st.session_state.get("patient_id")

if current_pid:
  history_df = get_patient_history(current_pid)

  if not history_df.empty:
    st.caption(
        f"Showing all recorded screening visits for Patient ID: `{current_pid}`"
    )

    # Clean display format
    display_df = history_df.copy()
    display_df["timestamp"] = pd.to_datetime(display_df["timestamp"]).dt.strftime(
        "%Y-%m-%d %H:%M"
    )

    st.dataframe(
        display_df[[
            "visit_id",
            "timestamp",
            "predicted_risk",
            "left_rom",
            "right_rom",
            "rom_asymmetry",
            "pain",
        ]],
        use_container_width=True,
    )

    # Plot ROM trajectory if patient has completed multiple checkups
    if len(history_df) > 1:
      st.write("**Range of Motion (ROM) Trend Across Visits:**")
      trend_df = (
          display_df.sort_values("timestamp")
          .set_index("timestamp")[["left_rom", "right_rom"]]
      )
      st.line_chart(trend_df)
  else:
    st.info("No previous screening records found for this patient.")

# ============================================================
# DISCLAIMER
# ============================================================

st.divider()

st.warning(
    """
### ⚠️ Important Disclaimer

KneeSense NER is a **prototype research and screening-support system**.

It does **not diagnose osteoarthritis**.


"OsteoX is a preliminary screening-support system and does not replace diagnostic clinical imaging or specialist consultation. "
"The machine-learning risk engine is trained on cohort data from the Osteoarthritis Initiative (OAI) incorporating joint kinematics and symptom grading. "
"Clinical evaluations must be corroborated by a registered medical practitioner."

"""
)

st.caption(
    "KneeSense NER • SIH26004 • Prototype"
)