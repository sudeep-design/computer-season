# calisthenics/pullup.py
import cv2
import mediapipe as mp
import time
from utils import landmark_to_point

def start_pullups(camera_index=0, bar_y_ratio=0.25):
    cap = cv2.VideoCapture(camera_index)
    mpPose = mp.solutions.pose
    pose = mpPose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6)
    draw = mp.solutions.drawing_utils

    counter = 0
    stage = "down"
    pTime = 0

    while True:
        success, img = cap.read()
        if not success:
            break
        h, w, _ = img.shape
        bar_y = int(h * bar_y_ratio)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(imgRGB)

        if results.pose_landmarks:
            nose_landmark = results.pose_landmarks.landmark[mpPose.PoseLandmark.NOSE.value]
            nose_px = int(nose_landmark.y * h)

            # strict pull-up logic: nose (chin) above bar counts as rep
            if nose_px < bar_y and stage == "down":
                counter += 1
                stage = "up"
            if nose_px > bar_y + 30:
                stage = "down"

            draw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

            cv2.putText(img, f'Pull-ups: {counter}', (30,60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0,255,0), 3)
            cv2.putText(img, f'NosY: {nose_px}', (30,100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        # bar line
        cv2.line(img, (0, bar_y), (w, bar_y), (255,255,0), 2)

        cTime = time.time()
        fps = 1/(cTime - pTime) if pTime else 0
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (10,30), cv2.FONT_HERSHEY_PLAIN, 1.5, (255,0,255), 2)

        cv2.imshow("Pull-Up Counter", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_pullups()
