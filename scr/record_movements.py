import cv2
import mediapipe as mp
import math
import time
import numpy as np

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


def calculate_angle(a, b, c):

    angle = math.degrees(
        math.atan2(c[1] - b[1], c[0] - b[0])
        - math.atan2(a[1] - b[1], a[0] - b[0])
    )

    angle = abs(angle)

    if angle > 180:
        angle = 360 - angle

    return angle


camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

angles = []

start_time = time.time()

print("Starting movement recording...")
print("Move your knee naturally.")
print("Recording for 10 seconds...")

while camera.isOpened():

    success, frame = camera.read()

    if not success:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks:

        mp_draw.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        landmarks = results.pose_landmarks.landmark

        hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
        ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]

        angle = calculate_angle(
            (hip.x, hip.y),
            (knee.x, knee.y),
            (ankle.x, ankle.y)
        )

        angles.append(angle)

        cv2.putText(
            frame,
            f"Knee Angle: {angle:.1f}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    elapsed = time.time() - start_time

    cv2.putText(
        frame,
        f"Time: {elapsed:.1f}s",
        (30, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.imshow("KneeSense - Movement Recording", frame)

    if elapsed >= 10:
        break

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()
pose.close()


# Analyze collected data

if len(angles) > 0:

    angles = np.array(angles)

    max_angle = np.max(angles)
    min_angle = np.min(angles)
    rom = max_angle - min_angle
    average_angle = np.mean(angles)

    print("\n========== KNEE MOVEMENT RESULTS ==========")

    print(f"Measurements      : {len(angles)}")
    print(f"Maximum angle     : {max_angle:.2f}°")
    print(f"Minimum angle     : {min_angle:.2f}°")
    print(f"Range of Motion   : {rom:.2f}°")
    print(f"Average angle     : {average_angle:.2f}°")

    print("============================================")

else:
    print("No pose data collected.")