# martial_arts/taekwondo.py
import cv2
import mediapipe as mp
import time
from utils import calculate_angle, landmark_to_point
import numpy as np

def start_taekwondo(camera_index=0, side='left'):
    """
    Detect front & side kicks on selected side ('left' or 'right').
    Uses knee angle (hip-knee-ankle). When knee goes from chamber (bent) -> extended (straight) count.
    """
    cap = cv2.VideoCapture(camera_index)
    mpPose = mp.solutions.pose
    pose = mpPose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    draw = mp.solutions.drawing_utils

    counter = 0
    stage = None
    pTime = 0

    side = side.lower()
    while True:
        success, img = cap.read()
        if not success:
            break
        h, w, _ = img.shape
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(imgRGB)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            if side == 'right':
                hip = landmark_to_point(lm[mpPose.PoseLandmark.RIGHT_HIP.value], w, h)
                knee = landmark_to_point(lm[mpPose.PoseLandmark.RIGHT_KNEE.value], w, h)
                ankle = landmark_to_point(lm[mpPose.PoseLandmark.RIGHT_ANKLE.value], w, h)
            else:
                hip = landmark_to_point(lm[mpPose.PoseLandmark.LEFT_HIP.value], w, h)
                knee = landmark_to_point(lm[mpPose.PoseLandmark.LEFT_KNEE.value], w, h)
                ankle = landmark_to_point(lm[mpPose.PoseLandmark.LEFT_ANKLE.value], w, h)

            angle = calculate_angle(hip, knee, ankle)  # knee angle

            # chamber when bent (< 90), extension when straight (>150)
            if angle < 90:
                stage = "chamber"
            if angle > 150 and stage == "chamber":
                counter += 1
                stage = "kick"

            # Visuals
            cv2.putText(img, f'Kicks: {counter}', (30,60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0,255,0), 3)
            cv2.putText(img, f'KneeAngle: {int(angle)}', (30,100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
            draw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
            # draw lines for leg
            cv2.line(img, hip, knee, (255,255,0), 2)
            cv2.line(img, knee, ankle, (255,255,0), 2)

        # FPS
        cTime = time.time()
        fps = 1/(cTime - pTime) if pTime else 0
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (10,30), cv2.FONT_HERSHEY_PLAIN, 1.5, (255,0,255), 2)

        cv2.imshow("Taekwondo Kicks", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_taekwondo()
