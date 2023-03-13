import cv2
import mediapipe as mp
import numpy as np
import time
import sys
from imutils.video import VideoStream
import argparse
import datetime
import imutils

sys.path.append('D:\\work\\braiven\\punching\\mediapipe\\pose_detect\\api')
from sendtodb.app import savedata

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

named_tuple = time.localtime() # get struct_time
time_string = time.strftime("%d/%m/%Y, %H:%M:%S", named_tuple)

timer = 0
start_time = time.time()
hand_up = False

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

cap = cv2.VideoCapture(0)
with mp_pose.Pose(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as pose:
    


    while cap.isOpened():
        ret, frame = cap.read()

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark
            # print(len(landmarks))
            l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            l_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            l_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            
            r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            r_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            r_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            r_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]


            l_angle = calculate_angle(l_hip,l_shoulder,l_elbow)
            r_angle = calculate_angle(r_hip,r_shoulder,r_elbow)
            # angle = calculate_angle(shoulder,elbow,wrist)
            # print(angle)

            # cv2.putText(image, str(angle), tuple(np.multiply(elbow, [640,482]).astype(int)),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,0,0),2,cv2.LINE_4)
            cv2.putText(image, str(l_angle), tuple(np.multiply(l_shoulder, [640,480]).astype(int)),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,0,0),2,cv2.LINE_4)
            cv2.putText(image, str(r_angle), tuple(np.multiply(r_shoulder, [640,480]).astype(int)),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,0,0),2,cv2.LINE_4)

            if l_angle > 63 or r_angle > 63:
                if hand_up != True:
                    hand_up = True
                    start_time = time.time()
                timer = time.time() - start_time
                print('timer1',timer)
                if timer > 3 and hand_up == True:
                    cv2.putText(image,'hand up',(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),cv2.LINE_4)
                    # data = {
                    #     'status' : 'hand up',
                    #     'time' : time_string
                    # }
                    # savedata().insert_data_mongo(data)
                    # print(l_angle, r_angle)
                    print(timer)
                    print(hand_up)
                    # hand_up = False

            if l_angle > 63 or r_angle > 63 and hand_up == True:

                timer = time.time() - start_time
                print('timer2',timer)
                

            # print(calculate_angle)

            # print('landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]',shoulder)
            # print('landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]',elbow)
            # print('landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]',wrist)
            # print(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value])
            # print('landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]',landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value])
            # print('landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]',landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])
        except:
            pass  
       

        # mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, 
        #                           mp_drawing.DrawingSpec(color = (245,117,66), thickness =2, circle_radius=2),
        #                           mp_drawing.DrawingSpec(color = (245,66,230), thickness =2, circle_radius=2))


        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()