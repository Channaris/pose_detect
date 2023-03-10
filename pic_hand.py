import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def calculate_angle(a,b,c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180:
        angle = 360-angle

    # print(angle)
    return angle

cap = cv2.imread('img/h.jpg')

with mp_pose.Pose( min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

    image = cv2.cvtColor(cap, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = pose.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    try:
        landmarks = results.pose_landmarks.landmark
        # print(len(landmarks))
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        
        angle = calculate_angle(hip,shoulder,elbow)
        # angle = calculate_angle(shoulder,elbow,wrist)
        print(angle)

        # cv2.putText(image, str(angle), tuple(np.multiply(elbow, [640,482]).astype(int)),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,0,0),2,cv2.LINE_4)
        cv2.putText(image, str(angle), tuple(np.multiply(shoulder, [640,480]).astype(int)),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,0,0),2,cv2.LINE_4)

        if angle > 63:
            cv2.putText(image,'hand up',(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),cv2.LINE_4)

        # print(calculate_angle)

        # print('landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]',shoulder)
        # print('landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]',elbow)
        # print('landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]',wrist)
        # print(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value])
        # print('landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]',landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value])
        # print('landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]',landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])
    except:
        pass  
    

    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, 
                                mp_drawing.DrawingSpec(color = (245,117,66), thickness =2, circle_radius=2),
                                mp_drawing.DrawingSpec(color = (245,66,230), thickness =2, circle_radius=2))

    cv2.imshow('MediaPipe Pose', image)
    cv2.waitKey(0)

cv2.destroyAllWindows()