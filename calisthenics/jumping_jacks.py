# calisthenics/jumping_jacks.py
import cv2
import mediapipe as mp
import time
import math

def start_jumping_jacks(camera_index=0):
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
            # normalized coords
            left_wrist_y = lm[mpPose.PoseLandmark.LEFT_WRIST.value].y
            right_wrist_y = lm[mpPose.PoseLandmark.RIGHT_WRIST.value].y
            left_ankle_x = lm[mpPose.PoseLandmark.LEFT_ANKLE.value].x
            right_ankle_x = lm[mpPose.PoseLandmark.RIGHT_ANKLE.value].x

            # hands up detection: wrists above head ≈ y < 0.35
            hands_up = left_wrist_y < 0.35 and right_wrist_y < 0.35
            # feet apart detection: ankle separation ratio > threshold
            ankle_sep = abs(left_ankle_x - right_ankle_x)

            if hands_up and ankle_sep > 0.25 and stage != "open":
                counter += 1
                stage = "open"
            if not hands_up and ankle_sep < 0.15:
                stage = "closed"

            draw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
            cv2.putText(img, f'Jumping Jacks: {counter}', (30,60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0,255,0), 3)
            cv2.putText(img, f'AnkleSep: {ankle_sep:.2f}', (30,100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

        cTime = time.time()
        fps = 1/(cTime - pTime) if pTime else 0
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (10,30), cv2.FONT_HERSHEY_PLAIN, 1.5, (255,0,255), 2)

        cv2.imshow("Jumping Jacks", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_jumping_jacks()
