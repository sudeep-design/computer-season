# calisthenics/pushup.py
import cv2
import mediapipe as mp
import time
from utils import calculate_angle, landmark_to_point
import numpy as np

def start_pushups(camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    mpPose = mp.solutions.pose
    pose = mpPose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    draw = mp.solutions.drawing_utils

    counter = 0
    stage = None
    pTime = 0

    while True:
        success, img = cap.read()
        if not success:
            break
        h, w, _ = img.shape
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(imgRGB)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            # using LEFT side; you can average both sides for robustness
            shoulder = landmark_to_point(lm[mpPose.PoseLandmark.LEFT_SHOULDER.value], w, h)
            elbow = landmark_to_point(lm[mpPose.PoseLandmark.LEFT_ELBOW.value], w, h)
            hip = landmark_to_point(lm[mpPose.PoseLandmark.LEFT_HIP.value], w, h)

            angle = calculate_angle(elbow, shoulder, hip)  # shoulder angle

            # Proper form thresholds & counting
            # DOWN when angle < 90; UP when angle > 160
            if angle > 160:
                if stage == "down":
                    counter += 1
                stage = "up"
            elif angle < 90:
                stage = "down"

            # Display
            cv2.putText(img, f'Push-ups: {counter}', (30,60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0,255,0), 3)
            cv2.putText(img, f'ShoulderAngle: {int(angle)}', (30,100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
            draw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

        # FPS
        cTime = time.time()
        fps = 1/(cTime - pTime) if pTime else 0
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (10,30), cv2.FONT_HERSHEY_PLAIN, 1.5, (255,0,255), 2)

        cv2.imshow("Push-Up Counter", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_pushups()
