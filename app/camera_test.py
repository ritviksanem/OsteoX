import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
import mediapipe as mp
import math
import time
import numpy as np
import pandas as pd
from pathlib import Path
import threading


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FEATURES_FOLDER = PROJECT_ROOT / "data" / "features"
FEATURES_FOLDER.mkdir(parents=True, exist_ok=True)

WALKING_FILE = FEATURES_FOLDER / "walking_features.csv"


# =========================================================
# PAGE
# =========================================================

st.set_page_config(
    page_title="KneeSense NER - Movement",
    page_icon="🦵",
    layout="wide"
)

st.title("🦵 KneeSense NER")
st.subheader("📷 Live Movement Assessment")

st.info(
    "Stand approximately 2–3 metres from the camera. "
    "Keep your full body visible and perform a natural walking "
    "or knee movement test."
)


# =========================================================
# ANGLE CALCULATION
# =========================================================

def calculate_angle(a, b, c):

    angle = math.degrees(
        math.atan2(c[1] - b[1], c[0] - b[0])
        -
        math.atan2(a[1] - b[1], a[0] - b[0])
    )

    angle = abs(angle)

    if angle > 180:
        angle = 360 - angle

    return angle


# =========================================================
# MEDIAPIPE PROCESSOR
# =========================================================

class KneeSenseProcessor(VideoProcessorBase):

    def __init__(self):

        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils

        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.recording = False
        self.start_time = None

        self.left_angles = []
        self.right_angles = []

        self.result = None

        self.lock = threading.Lock()

    # -----------------------------------------------------
    # START TEST
    # -----------------------------------------------------

    def start_test(self):

        with self.lock:

            self.recording = True
            self.start_time = time.time()

            self.left_angles = []
            self.right_angles = []

            self.result = None

    # -----------------------------------------------------
    # PROCESS CAMERA FRAME
    # -----------------------------------------------------

    def recv(self, frame):

        image = frame.to_ndarray(format="bgr24")

        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        results = self.pose.process(rgb)

        left_angle = None
        right_angle = None

        if results.pose_landmarks:

            landmarks = results.pose_landmarks.landmark

            # LEFT
            left_hip = landmarks[
                self.mp_pose.PoseLandmark.LEFT_HIP
            ]

            left_knee = landmarks[
                self.mp_pose.PoseLandmark.LEFT_KNEE
            ]

            left_ankle = landmarks[
                self.mp_pose.PoseLandmark.LEFT_ANKLE
            ]

            # RIGHT
            right_hip = landmarks[
                self.mp_pose.PoseLandmark.RIGHT_HIP
            ]

            right_knee = landmarks[
                self.mp_pose.PoseLandmark.RIGHT_KNEE
            ]

            right_ankle = landmarks[
                self.mp_pose.PoseLandmark.RIGHT_ANKLE
            ]

            left_angle = calculate_angle(
                (left_hip.x, left_hip.y),
                (left_knee.x, left_knee.y),
                (left_ankle.x, left_ankle.y)
            )

            right_angle = calculate_angle(
                (right_hip.x, right_hip.y),
                (right_knee.x, right_knee.y),
                (right_ankle.x, right_ankle.y)
            )

            # Draw skeleton
            self.mp_draw.draw_landmarks(
                image,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )

            # ------------------------------------------------
            # RECORD DATA
            # ------------------------------------------------

            with self.lock:

                if self.recording:

                    self.left_angles.append(left_angle)
                    self.right_angles.append(right_angle)

                    elapsed = time.time() - self.start_time

                    if elapsed >= 30:

                        self.recording = False

                        self.result = self.calculate_results()

        # =================================================
        # CAMERA OVERLAY
        # =================================================

        cv2.rectangle(
            image,
            (10, 10),
            (450, 145),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            image,
            "KneeSense NER",
            (25, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2
        )

        if left_angle is not None:

            cv2.putText(
                image,
                f"Left Knee: {left_angle:.1f} deg",
                (25, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

        if right_angle is not None:

            cv2.putText(
                image,
                f"Right Knee: {right_angle:.1f} deg",
                (25, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

        if self.recording:

            elapsed = time.time() - self.start_time
            remaining = max(0, 30 - elapsed)

            cv2.putText(
                image,
                f"RECORDING: {remaining:.1f}s",
                (25, 135),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

        else:

            cv2.putText(
                image,
                "READY",
                (25, 135),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

        return av.VideoFrame.from_ndarray(
            image,
            format="bgr24"
        )

    # =====================================================
    # CALCULATE FEATURES
    # =====================================================

    def calculate_results(self):

        left = np.array(self.left_angles)
        right = np.array(self.right_angles)

        if len(left) < 20 or len(right) < 20:

            return {
                "error": "Not enough valid movement data."
            }

        left_max = np.max(left)
        left_min = np.min(left)
        left_rom = left_max - left_min
        left_avg = np.mean(left)

        right_max = np.max(right)
        right_min = np.min(right)
        right_rom = right_max - right_min
        right_avg = np.mean(right)

        # ROM asymmetry
        average_rom = (left_rom + right_rom) / 2

        if average_rom > 0:

            asymmetry = (
                abs(left_rom - right_rom)
                / average_rom
            ) * 100

        else:

            asymmetry = 0

        # Movement consistency
        left_std = np.std(left)
        right_std = np.std(right)

        left_consistency = max(
            0,
            min(
                100,
                100 - (left_std / 2)
            )
        )

        right_consistency = max(
            0,
            min(
                100,
                100 - (right_std / 2)
            )
        )

        valid_samples = min(
            len(left),
            len(right)
        )

        result = {
            "left_max_angle": left_max,
            "left_min_angle": left_min,
            "left_rom": left_rom,
            "left_average_angle": left_avg,

            "right_max_angle": right_max,
            "right_min_angle": right_min,
            "right_rom": right_rom,
            "right_average_angle": right_avg,

            "rom_asymmetry": asymmetry,

            "left_consistency": left_consistency,
            "right_consistency": right_consistency,

            "valid_samples": valid_samples,

            "sample_rate": valid_samples / 30
        }

        return result


# =========================================================
# CAMERA
# =========================================================

ctx = webrtc_streamer(
    key="kneesense-live-camera",

    video_processor_factory=KneeSenseProcessor,

    media_stream_constraints={
        "video": True,
        "audio": False
    },

    async_processing=True
)


# =========================================================
# CONTROL
# =========================================================

st.divider()

st.subheader("🎬 Movement Test")

if ctx.state.playing:

    st.success("Camera connected.")

    if st.button(
        "▶️ START 30-SECOND TEST",
        use_container_width=True
    ):

        if ctx.video_processor is not None:

            ctx.video_processor.start_test()

            st.success(
                "Movement test started. "
                "Continue moving for 30 seconds."
            )

else:

    st.warning(
        "Start the camera above before beginning the test."
    )


# =========================================================
# RESULTS
# =========================================================

if (
    ctx.video_processor is not None
    and ctx.video_processor.result is not None
):

    result = ctx.video_processor.result

    st.divider()

    st.header("📊 Movement Analysis")

    if "error" in result:

        st.error(result["error"])

    else:

        # Save features
        pd.DataFrame([result]).to_csv(
            WALKING_FILE,
            index=False
        )

        st.success(
            "✅ 30-second movement analysis completed."
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Left Knee ROM",
                f'{result["left_rom"]:.2f}°'
            )

        with col2:

            st.metric(
                "Right Knee ROM",
                f'{result["right_rom"]:.2f}°'
            )

        with col3:

            st.metric(
                "ROM Asymmetry",
                f'{result["rom_asymmetry"]:.2f}%'
            )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Left Consistency",
                f'{result["left_consistency"]:.2f}%'
            )

        with col2:

            st.metric(
                "Right Consistency",
                f'{result["right_consistency"]:.2f}%'
            )

        st.metric(
            "Valid Samples",
            result["valid_samples"]
        )

        st.success(
            f"Movement features saved to:\n{WALKING_FILE}"
        )


# =========================================================
# DISCLAIMER
# =========================================================

st.divider()

st.warning(
    "KneeSense NER is a preliminary screening-support "
    "prototype and NOT a medical diagnosis."
)