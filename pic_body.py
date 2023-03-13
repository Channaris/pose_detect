import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180:
        angle = 360-angle
        print(angle)

    return angle


cap = cv2.imread('img/l.jpg')
# cap = cv2.imread('img/s.png')

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

    image = cv2.cvtColor(cap, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = pose.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    try:
        landmarks = results.pose_landmarks.landmark
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y

        print(left_hip, left_shoulder, right_hip, right_shoulder)

        image.flags.writeable = True

        print(left_hip - left_shoulder, right_hip - right_shoulder)

        if -0.01 <= left_hip - left_shoulder <= 0.01 or -0.01 <= right_hip - right_shoulder <= 0.01:
            cv2.putText(image, 'lying down', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), cv2.LINE_4)

    except:
        pass

    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

    cv2.imshow('MediaPipe Pose', image)
    cv2.waitKey(0)

cv2.destroyAllWindows()
