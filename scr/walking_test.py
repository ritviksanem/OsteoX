import cv2
import mediapipe as mp
import math
import time
import numpy as np
import pandas as pd

from pathlib import Path
from scipy.signal import find_peaks


# ============================================================
# PROJECT PATHS
# ============================================================

project_root = Path(__file__).resolve().parent.parent

features_folder = project_root / "data" / "features"
processed_folder = project_root / "data" / "processed"

features_folder.mkdir(parents=True, exist_ok=True)
processed_folder.mkdir(parents=True, exist_ok=True)


# ============================================================
# MEDIAPIPE
# ============================================================

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# ============================================================
# 3D KNEE ANGLE
# ============================================================

def calculate_3d_angle(a, b, c):

    a = np.array([a.x, a.y, a.z])
    b = np.array([b.x, b.y, b.z])
    c = np.array([c.x, c.y, c.z])

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (
        np.linalg.norm(ba) * np.linalg.norm(bc)
    )

    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    angle = np.degrees(np.arccos(cosine_angle))

    return angle


# ============================================================
# SMOOTH SIGNAL
# ============================================================

def smooth_signal(signal, window=7):

    if len(signal) < window:
        return np.array(signal)

    return (
        pd.Series(signal)
        .rolling(window=window, center=True)
        .median()
        .bfill()
        .ffill()
        .values
    )


# ============================================================
# DETECT GAIT CYCLES
# ============================================================

def detect_gait_cycles(signal, times):

    signal = np.array(signal)

    if len(signal) < 10:
        return [], []

    # Find valleys by finding peaks in the negative signal
    valleys, properties = find_peaks(
        -signal,

        # At least ~0.75 sec between events
        distance=12,

        # Ignore very small fluctuations
        prominence=4
    )

    # Keep only reasonable knee flexion events
    valid_valleys = []

    for index in valleys:

        angle = signal[index]

        if 130 <= angle <= 175:
            valid_valleys.append(index)

    # Calculate time between same-leg events
    cycle_times = []

    for i in range(1, len(valid_valleys)):

        previous = valid_valleys[i - 1]
        current = valid_valleys[i]

        cycle_time = times[current] - times[previous]

        # Reasonable human walking cycle range
        if 0.75 <= cycle_time <= 2.5:
            cycle_times.append(cycle_time)

    return valid_valleys, cycle_times


# ============================================================
# CAMERA
# ============================================================

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    print("ERROR: Camera could not be opened.")
    exit()


# ============================================================
# DATA STORAGE
# ============================================================

times = []

left_angles = []
right_angles = []

start_time = time.perf_counter()

test_duration = 30


print("\n======================================")
print("       KNEESENSE WALKING TEST")
print("======================================")
print("\nInstructions:")
print("1. Keep your full body visible.")
print("2. Walk naturally.")
print("3. Walk left-to-right across the camera.")
print("4. Keep a comfortable walking speed.")
print("5. Test duration: 30 seconds.")
print("\nStarting test...\n")


# ============================================================
# RECORDING
# ============================================================

while camera.isOpened():

    success, frame = camera.read()

    if not success:
        print("Camera frame could not be read.")
        break

    current_time = time.perf_counter() - start_time

    if current_time >= test_duration:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = pose.process(rgb)

    if results.pose_landmarks and results.pose_world_landmarks:

        landmarks_2d = results.pose_landmarks.landmark
        landmarks_3d = results.pose_world_landmarks.landmark

        # Check visibility
        required_points = [
            mp_pose.PoseLandmark.LEFT_HIP,
            mp_pose.PoseLandmark.LEFT_KNEE,
            mp_pose.PoseLandmark.LEFT_ANKLE,
            mp_pose.PoseLandmark.RIGHT_HIP,
            mp_pose.PoseLandmark.RIGHT_KNEE,
            mp_pose.PoseLandmark.RIGHT_ANKLE
        ]

        visible = True

        for point in required_points:

            if landmarks_2d[point].visibility < 0.5:
                visible = False
                break

        if visible:

            # LEFT LEG
            left_hip = landmarks_3d[
                mp_pose.PoseLandmark.LEFT_HIP
            ]

            left_knee = landmarks_3d[
                mp_pose.PoseLandmark.LEFT_KNEE
            ]

            left_ankle = landmarks_3d[
                mp_pose.PoseLandmark.LEFT_ANKLE
            ]

            # RIGHT LEG
            right_hip = landmarks_3d[
                mp_pose.PoseLandmark.RIGHT_HIP
            ]

            right_knee = landmarks_3d[
                mp_pose.PoseLandmark.RIGHT_KNEE
            ]

            right_ankle = landmarks_3d[
                mp_pose.PoseLandmark.RIGHT_ANKLE
            ]

            # Calculate angles
            left_angle = calculate_3d_angle(
                left_hip,
                left_knee,
                left_ankle
            )

            right_angle = calculate_3d_angle(
                right_hip,
                right_knee,
                right_ankle
            )

            # Reject impossible values
            if (
                30 <= left_angle <= 180
                and
                30 <= right_angle <= 180
            ):

                times.append(current_time)

                left_angles.append(left_angle)
                right_angles.append(right_angle)

                # Display angles
                cv2.putText(
                    frame,
                    f"Left Knee: {left_angle:.1f}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"Right Knee: {right_angle:.1f}",
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

    # Draw pose
    if results.pose_landmarks:

        mp_draw.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

    # Timer
    remaining = max(
        0,
        int(test_duration - current_time)
    )

    cv2.putText(
        frame,
        f"Time: {remaining}s",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "KneeSense - Walking Test",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ============================================================
# CLEANUP
# ============================================================

camera.release()
cv2.destroyAllWindows()
pose.close()


# ============================================================
# CHECK DATA
# ============================================================

if len(left_angles) < 20:

    print("\nERROR: Not enough valid pose data.")
    print("Try improving lighting and camera positioning.")
    exit()


# ============================================================
# SMOOTH ANGLES
# ============================================================

left_smoothed = smooth_signal(
    left_angles,
    window=7
)

right_smoothed = smooth_signal(
    right_angles,
    window=7
)


# ============================================================
# GAIT EVENT DETECTION
# ============================================================

left_events, left_cycle_times = detect_gait_cycles(
    left_smoothed,
    times
)

right_events, right_cycle_times = detect_gait_cycles(
    right_smoothed,
    times
)


# ============================================================
# BASIC FEATURES
# ============================================================

left_max = np.max(left_smoothed)
left_min = np.min(left_smoothed)
left_rom = left_max - left_min
left_avg = np.mean(left_smoothed)

right_max = np.max(right_smoothed)
right_min = np.min(right_smoothed)
right_rom = right_max - right_min
right_avg = np.mean(right_smoothed)


# ============================================================
# CYCLE TIME
# ============================================================

left_avg_cycle = (
    np.mean(left_cycle_times)
    if len(left_cycle_times) > 0
    else 0
)

right_avg_cycle = (
    np.mean(right_cycle_times)
    if len(right_cycle_times) > 0
    else 0
)


# ============================================================
# CADENCE
# ============================================================

duration = times[-1] - times[0]

# Same-leg cycles × 2 = approximate steps
left_cadence = (
    len(left_cycle_times) * 2 / duration * 60
    if duration > 0
    else 0
)

right_cadence = (
    len(right_cycle_times) * 2 / duration * 60
    if duration > 0
    else 0
)


# ============================================================
# CONSISTENCY
# ============================================================

def calculate_consistency(cycle_times):

    if len(cycle_times) < 2:
        return 0

    mean_cycle = np.mean(cycle_times)

    if mean_cycle == 0:
        return 0

    cv = np.std(cycle_times) / mean_cycle

    score = max(
        0,
        100 - (cv * 100)
    )

    return score


left_consistency = calculate_consistency(
    left_cycle_times
)

right_consistency = calculate_consistency(
    right_cycle_times
)


# ============================================================
# ROM ASYMMETRY
# ============================================================

largest_rom = max(
    left_rom,
    right_rom
)

if largest_rom > 0:

    rom_asymmetry = (
        abs(left_rom - right_rom)
        / largest_rom
        * 100
    )

else:

    rom_asymmetry = 0


# ============================================================
# DATA QUALITY
# ============================================================

valid_samples = len(times)

sample_rate = (
    valid_samples / duration
    if duration > 0
    else 0
)

if sample_rate >= 12:

    quality = "GOOD"

elif sample_rate >= 8:

    quality = "ACCEPTABLE"

else:

    quality = "LOW"


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n======================================")
print("        WALKING TEST RESULTS")
print("======================================")

print("\nLEFT KNEE")

print(
    f"Maximum Angle       : {left_max:.2f}°"
)

print(
    f"Minimum Angle       : {left_min:.2f}°"
)

print(
    f"ROM                 : {left_rom:.2f}°"
)

print(
    f"Average Angle       : {left_avg:.2f}°"
)

print(
    f"Gait Events         : {len(left_events)}"
)

print(
    f"Average Cycle Time  : {left_avg_cycle:.2f}s"
)

print(
    f"Approx Cadence      : {left_cadence:.2f} steps/min"
)

print(
    f"Consistency         : {left_consistency:.2f}%"
)


print("\nRIGHT KNEE")

print(
    f"Maximum Angle       : {right_max:.2f}°"
)

print(
    f"Minimum Angle       : {right_min:.2f}°"
)

print(
    f"ROM                 : {right_rom:.2f}°"
)

print(
    f"Average Angle       : {right_avg:.2f}°"
)

print(
    f"Gait Events         : {len(right_events)}"
)

print(
    f"Average Cycle Time  : {right_avg_cycle:.2f}s"
)

print(
    f"Approx Cadence      : {right_cadence:.2f} steps/min"
)

print(
    f"Consistency         : {right_consistency:.2f}%"
)


print("\nSYMMETRY")

print(
    f"ROM Asymmetry       : {rom_asymmetry:.2f}%"
)


print("\nDATA QUALITY")

print(
    f"Valid Samples       : {valid_samples}"
)

print(
    f"Sample Rate         : {sample_rate:.2f} frames/sec"
)

print(
    f"Quality             : {quality}"
)


# ============================================================
# SAVE FEATURES
# ============================================================

features = {

    "left_max_angle": left_max,
    "left_min_angle": left_min,
    "left_rom": left_rom,
    "left_average_angle": left_avg,

    "right_max_angle": right_max,
    "right_min_angle": right_min,
    "right_rom": right_rom,
    "right_average_angle": right_avg,

    "rom_asymmetry": rom_asymmetry,

    "left_gait_events": len(left_events),
    "right_gait_events": len(right_events),

    "left_average_cycle_time": left_avg_cycle,
    "right_average_cycle_time": right_avg_cycle,

    "left_cadence": left_cadence,
    "right_cadence": right_cadence,

    "left_consistency": left_consistency,
    "right_consistency": right_consistency,

    "valid_samples": valid_samples,
    "sample_rate": sample_rate
}


features_df = pd.DataFrame(
    [features]
)

features_file = (
    features_folder /
    "walking_features.csv"
)

features_df.to_csv(
    features_file,
    index=False
)


# ============================================================
# SAVE RAW + SMOOTHED DATA
# ============================================================

raw_df = pd.DataFrame({

    "time": times,

    "left_knee_angle_raw": left_angles,

    "right_knee_angle_raw": right_angles,

    "left_knee_angle_smoothed": left_smoothed,

    "right_knee_angle_smoothed": right_smoothed
})


raw_file = (
    processed_folder /
    "walking_raw.csv"
)

raw_df.to_csv(
    raw_file,
    index=False
)


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n======================================")

print(
    f"Walking features saved to:\n{features_file}"
)

print(
    f"\nRaw walking data saved to:\n{raw_file}"
)

print("\n======================================")
print("             TEST COMPLETE")
print("======================================")