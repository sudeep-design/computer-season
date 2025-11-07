# martial_arts/karate.py
import cv2
import mediapipe as mp
import time
from utils import calculate_angle, landmark_to_point

def start_karate(camera_index=0, side='left'):
    """
    Detect straight punches (jab/reverse) using elbow angle and forward displacement.
    side: 'left' or 'right'
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
                shoulder = landmark_to_point(lm[mpPose.PoseLandmark.RIGHT_SHOULDER.value], w, h)
                elbow = landmark_to_point(lm[mpPose.PoseLandmark.RIGHT_ELBOW.value], w, h)
                wrist = landmark_to_point(lm[mpPose.PoseLandmark.RIGHT_WRIST.value], w, h)
            else:
                shoulder = landmark_to_point(lm[mpPose.PoseLandmark.LEFT_SHOULDER.value], w, h)
                elbow = landmark_to_point(lm[mpPose.PoseLandmark.LEFT_ELBOW.value], w, h)
                wrist = landmark_to_point(lm[mpPose.PoseLandmark.LEFT_WRIST.value], w, h)

            angle = calculate_angle(shoulder, elbow, wrist)  # elbow angle
            # forward displacement: how far wrist x passes shoulder x (positive when forward)
            forward_disp = (shoulder[0] - wrist[0]) if side == 'right' else (wrist[0] - shoulder[0])
            disp_thresh = int(0.06 * w)  # ~6% of width

            # loading (elbow bent) and punch (arm straight + forward)
            if angle < 70:
                stage = "loaded"
            if angle > 160 and stage == "loaded" and forward_disp > disp_thresh:
                counter += 1
                stage = "punched"

            cv2.putText(img, f'Punches: {counter}', (30,60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0,255,0), 3)
            cv2.putText(img, f'ElbowAngle: {int(angle)}', (30,100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
            cv2.putText(img, f'Disp: {int(forward_disp)}', (30,130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
            draw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
            cv2.line(img, shoulder, elbow, (255,255,0), 2)
            cv2.line(img, elbow, wrist, (255,255,0), 2)

        # FPS
        cTime = time.time()
        fps = 1/(cTime - pTime) if pTime else 0
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (10,30), cv2.FONT_HERSHEY_PLAIN, 1.5, (255,0,255), 2)

        cv2.imshow("Karate Punches", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_karate()
