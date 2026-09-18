import cv2
import mediapipe as mp
import math

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


def calculate_angle(a, b, c):
    """
    Calculate angle ABC
    a = hip
    b = knee
    c = ankle
    """

    angle = math.degrees(
        math.atan2(c[1] - b[1], c[0] - b[0])
        - math.atan2(a[1] - b[1], a[0] - b[0])
    )

    angle = abs(angle)

    if angle > 180:
        angle = 360 - angle

    return angle


camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while camera.isOpened():

    success, frame = camera.read()

    if not success:
        print("Camera error")
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

        # LEFT LEG
        hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
        ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]

        hip_point = (hip.x, hip.y)
        knee_point = (knee.x, knee.y)
        ankle_point = (ankle.x, ankle.y)

        knee_angle = calculate_angle(
            hip_point,
            knee_point,
            ankle_point
        )

        # Display angle
        cv2.putText(
            frame,
            f"Left Knee: {knee_angle:.1f} deg",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        print(f"Left Knee Angle: {knee_angle:.1f}")

    cv2.imshow(
        "KneeSense - Knee Angle",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()
pose.close()