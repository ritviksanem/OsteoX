import cv2
import mediapipe as mp
import math
import csv
from pathlib import Path


# -----------------------------
# Calculate angle
# -----------------------------
def calculate_angle(a, b, c):

    angle = math.degrees(
        math.atan2(c[1] - b[1], c[0] - b[0])
        - math.atan2(a[1] - b[1], a[0] - b[0])
    )

    angle = abs(angle)

    if angle > 180:
        angle = 360 - angle

    return angle


# -----------------------------
# MediaPipe setup
# -----------------------------
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# -----------------------------
# Camera
# -----------------------------
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    print("❌ Camera not found")
    exit()


# -----------------------------
# Data storage
# -----------------------------
left_angles = []
right_angles = []

left_bent = False
right_bent = False

left_repetitions = 0
right_repetitions = 0

start_time = cv2.getTickCount()
duration = 15


print("\n===================================")
print("      KneeSense - Step 5")
print("   Both Knee Symmetry Analysis")
print("===================================")
print("\nStand where your full body is visible.")
print("Perform slow knee bending movements.")
print("Recording for 15 seconds...\n")


# -----------------------------
# Main loop
# -----------------------------
while camera.isOpened():

    success, frame = camera.read()

    if not success:
        print("❌ Camera frame not received")
        break

    # Convert BGR → RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = pose.process(rgb)

    current_time = cv2.getTickCount()
    elapsed = (current_time - start_time) / cv2.getTickFrequency()

    # Stop after 15 seconds
    if elapsed >= duration:
        break

    if results.pose_landmarks:

        landmarks = results.pose_landmarks.landmark

        # -----------------------------
        # LEFT LEG
        # -----------------------------
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
        left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]

        left_angle = calculate_angle(
            (left_hip.x, left_hip.y),
            (left_knee.x, left_knee.y),
            (left_ankle.x, left_ankle.y)
        )

        # -----------------------------
        # RIGHT LEG
        # -----------------------------
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
        right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
        right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]

        right_angle = calculate_angle(
            (right_hip.x, right_hip.y),
            (right_knee.x, right_knee.y),
            (right_ankle.x, right_ankle.y)
        )

        # Store angles
        left_angles.append(left_angle)
        right_angles.append(right_angle)

        # -----------------------------
        # LEFT REPETITION
        # -----------------------------
        if left_angle < 120:
            left_bent = True

        if left_angle > 160 and left_bent:
            left_repetitions += 1
            left_bent = False

        # -----------------------------
        # RIGHT REPETITION
        # -----------------------------
        if right_angle < 120:
            right_bent = True

        if right_angle > 160 and right_bent:
            right_repetitions += 1
            right_bent = False

        # -----------------------------
        # Draw pose
        # -----------------------------
        mp_draw.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

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
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Left Reps: {left_repetitions}",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Right Reps: {right_repetitions}",
            (20, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        remaining = int(duration - elapsed)

        cv2.putText(
            frame,
            f"Time: {remaining}s",
            (20, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

    cv2.imshow("KneeSense - Both Knee Analysis", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# -----------------------------
# Close camera
# -----------------------------
camera.release()
cv2.destroyAllWindows()
pose.close()


# -----------------------------
# Calculate features
# -----------------------------
if len(left_angles) == 0 or len(right_angles) == 0:

    print("\n❌ No knee movement data recorded.")
    exit()


left_max = max(left_angles)
left_min = min(left_angles)
left_rom = left_max - left_min
left_average = sum(left_angles) / len(left_angles)

right_max = max(right_angles)
right_min = min(right_angles)
right_rom = right_max - right_min
right_average = sum(right_angles) / len(right_angles)


# -----------------------------
# Calculate asymmetry
# -----------------------------
maximum_rom = max(left_rom, right_rom)

if maximum_rom > 0:

    asymmetry = (
        abs(left_rom - right_rom)
        / maximum_rom
    ) * 100

else:

    asymmetry = 0


# -----------------------------
# Display results
# -----------------------------
print("\n===================================")
print("         KNEESENSE RESULTS")
print("===================================")

print("\nLEFT KNEE")
print(f"Maximum Angle : {left_max:.2f}°")
print(f"Minimum Angle : {left_min:.2f}°")
print(f"ROM           : {left_rom:.2f}°")
print(f"Average Angle : {left_average:.2f}°")
print(f"Repetitions   : {left_repetitions}")

print("\nRIGHT KNEE")
print(f"Maximum Angle : {right_max:.2f}°")
print(f"Minimum Angle : {right_min:.2f}°")
print(f"ROM           : {right_rom:.2f}°")
print(f"Average Angle : {right_average:.2f}°")
print(f"Repetitions   : {right_repetitions}")

print("\nSYMMETRY")
print(f"ROM Asymmetry : {asymmetry:.2f}%")

print("===================================")


# -----------------------------
# Save CSV
# -----------------------------
project_root = Path(__file__).resolve().parent.parent

features_folder = project_root / "data" / "features"

features_folder.mkdir(
    parents=True,
    exist_ok=True
)

file_path = features_folder / "knee_features.csv"


with open(file_path, "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow([
        "left_repetitions",
        "left_max_angle",
        "left_min_angle",
        "left_rom",
        "left_average_angle",

        "right_repetitions",
        "right_max_angle",
        "right_min_angle",
        "right_rom",
        "right_average_angle",

        "rom_asymmetry"
    ])

    writer.writerow([
        left_repetitions,
        round(left_max, 2),
        round(left_min, 2),
        round(left_rom, 2),
        round(left_average, 2),

        right_repetitions,
        round(right_max, 2),
        round(right_min, 2),
        round(right_rom, 2),
        round(right_average, 2),

        round(asymmetry, 2)
    ])


print("\n✅ Features saved to:")
print(file_path)