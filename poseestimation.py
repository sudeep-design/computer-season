import cv2
import mediapipe as mp
import time
import math
import numpy as np

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0])
    angle = abs(radians*180.0/math.pi)
    if angle > 180:
        angle = 360 - angle
    return angle

cap = cv2.VideoCapture(0)
mpPose = mp.solutions.pose
pose = mpPose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils
pTime = 0

counter = 0
stage = None  # "up" or "down"

while True:
    success, img = cap.read()
    if not success:
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        shoulder = [landmarks[mpPose.PoseLandmark.LEFT_SHOULDER.value].x,
                    landmarks[mpPose.PoseLandmark.LEFT_SHOULDER.value].y]
        elbow = [landmarks[mpPose.PoseLandmark.LEFT_ELBOW.value].x,
                 landmarks[mpPose.PoseLandmark.LEFT_ELBOW.value].y]
        hip = [landmarks[mpPose.PoseLandmark.LEFT_HIP.value].x,
               landmarks[mpPose.PoseLandmark.LEFT_HIP.value].y]

        h, w, _ = img.shape
        shoulder = tuple(np.multiply(shoulder, [w, h]).astype(int))
        elbow = tuple(np.multiply(elbow, [w, h]).astype(int))
        hip = tuple(np.multiply(hip, [w, h]).astype(int))

        # Draw joints
        cv2.circle(img, shoulder, 5, (255,0,0), -1)
        cv2.circle(img, elbow, 5, (0,255,0), -1)
        cv2.circle(img, hip, 5, (0,0,255), -1)
        cv2.line(img, shoulder, elbow, (255,255,0), 2)
        cv2.line(img, elbow, hip, (255,255,0), 2)

        angle = calculate_angle(elbow, shoulder, hip)

        # ---------- Correct Counting Logic ----------
        if angle > 160:
            if stage == "down":
                counter += 1
            stage = "up"
        elif angle < 90:
            stage = "down"

        # Display reps and angle
        cv2.putText(img, f'Reps: {counter}', (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 3)
        cv2.putText(img, f'Angle: {int(angle)}', (50,150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 2)

        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

    # FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (10,50), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,255), 2)

    cv2.imshow("Push-Up Counter", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
