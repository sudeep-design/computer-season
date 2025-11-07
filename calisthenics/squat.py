# calisthenics/squat.py
import cv2
import mediapipe as mp
import time
from utils import calculate_angle, landmark_to_point

def start_squats(camera_index=0):
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
            # use RIGHT knee for stability (you can average both)
            hip = landmark_to_point(lm[mpPose.PoseLandmark.RIGHT_HIP.value], w, h)
            knee = landmark_to_point(lm[mpPose.PoseLandmark.RIGHT_KNEE.value], w, h)
            ankle = landmark_to_point(lm[mpPose.PoseLandmark.RIGHT_ANKLE.value], w, h)

            angle = calculate_angle(hip, knee, ankle)  # knee angle

            # Proper form thresholds: bottom < 70 (deep), standing > 160
            if angle > 160:
                if stage == "down":
                    counter += 1
                stage = "up"
            elif angle < 90:
                stage = "down"

            cv2.putText(img, f'Squats: {counter}', (30,60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0,255,0), 3)
            cv2.putText(img, f'KneeAngle: {int(angle)}', (30,100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
            draw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

        # FPS
        cTime = time.time()
        fps = 1/(cTime - pTime) if pTime else 0
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (10,30), cv2.FONT_HERSHEY_PLAIN, 1.5, (255,0,255), 2)

        cv2.imshow("Squat Counter", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_squats()
