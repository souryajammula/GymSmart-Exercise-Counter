#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 15:40:29 2022

@author: sahibbhatia
"""

import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time
import PoseModel as pm

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
#Add an expander to the app 

st.header("GymSmart Exercise App")
    
count = 0
runT = st.button('Start Personal Training')
#Adding a sidebar to the app
st.sidebar.title("Count your reps!")

run = st.button('Start Counter')
with st.expander("How to use"):
        st.write("""
         1. Click the button above to start personal training.
         2. 0 and 100 define the start and end position of exercise
         3. Stop training on top right.\n
         
         To Only count the reps:
             
         1. Click the start counter button to start the app.
         2. Set the camera to see arms and chest.
         3. Start exercising and look at your posture on the screen.
         4. Reset counter for next rep.
         5. Click stop button to stop.
        """)

#run = st.sidebar.checkbox('Check to start and uncheck to stop!')
st.sidebar.text("Please put your hands in proper \nposition for exercise and start \nthe counter!\nIf you need help in doing an \nexercise, read how to use and start \npersonal trainer.")
FRAME_WINDOW = st.image([])

# Calculate Angles

def calculate_angle(a,b,c):
    # Reduce 3D point to 2D
    a = np.array([a.x, a.y])#, a.z])    
    b = np.array([b.x, b.y])#, b.z])
    c = np.array([c.x, c.y])#, c.z])  

    ab = np.subtract(a, b)
    bc = np.subtract(b, c)
    
    # A.B = |A||B|cos(x) 
    theta = np.arccos(np.dot(ab, bc) / np.multiply(np.linalg.norm(ab), np.linalg.norm(bc)))     
    # Convert to degrees
    theta = 180 - 180 * theta / 3.14   
    return np.round(theta, 2)


def curlCOunter():
    # Connecting Keypoints Visuals
    mp_drawing = mp.solutions.drawing_utils     

    # Keypoint detection model
    mp_pose = mp.solutions.pose     

    # Flag which stores hand position(Either UP or DOWN)
    left_flag = None     
    right_flag = None

    # Storage for count of bicep curls
    right_count = 0
    left_count = 0       

    cap = cv2.VideoCapture(0)
    # Landmark detection model instance
    pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.5) 
    while cap.isOpened():
        _, frame = cap.read()

         # Convert BGR frame to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)     
        image.flags.writeable = False
        
        # Make Detections
        # Get landmarks of the object in frame from the model
        results = pose.process(image)   

        # Back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)      

        try:
            # Extract Landmarks
            landmarks = results.pose_landmarks.landmark

            # Get coordinates of left part
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
            left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]

            # Get coordinates of right part
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
            right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]

            # Calculate and get angle
            left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)      
            right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

            # Visualize angle
            cv2.putText(image,\
                    str(left_angle), \
                        tuple(np.multiply([left_elbow.x, left_elbow.y], [640,480]).astype(int)),\
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),2,cv2.LINE_AA)
            cv2.putText(image,\
                    str(right_angle), \
                        tuple(np.multiply([right_elbow.x, right_elbow.y], [640,480]).astype(int)),\
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),2,cv2.LINE_AA)
        
            # Counter 
            if left_angle > 160:
                left_flag = 'down'
            if left_angle < 50 and left_flag=='down':
                left_count += 1
                left_flag = 'up'

            if right_angle > 160:
                right_flag = 'down'
            if right_angle < 50 and right_flag=='down':
                right_count += 1
                right_flag = 'up'
            
        except:
            pass

        # Setup Status Box
        cv2.rectangle(image, (0,0), (1024,73), (10,10,10), -1)
        cv2.putText(image, 'Left=' + str(left_count) + '    Right=' + str(right_count),
                          (10,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        
        # Render Detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #cv2.imshow('MediaPipe feed', image)

        FRAME_WINDOW.image(image)

        # Esc for quiting the app
        k = cv2.waitKey(30) & 0xff  
        if k==27:
            break
        elif k==ord('r'):       
            # Reset the counter on pressing 'r' on the Keyboard
            left_count = 0
            right_count = 0

    cap.release()
    cv2.destroyAllWindows()


def startTrainer():
    cap = cv2.VideoCapture(0)
     
    detector = pm.poseDetector()
    count = 0
    dir = 0
    pTime = 0
    while True:
        success, img = cap.read()
        img = cv2.resize(img, (1280, 720))
        # img = cv2.imread("AiTrainer/test.jpg")
        img = detector.findPose(img, False)
        lmList = detector.findPosition(img, False)
        # print(lmList)
        if len(lmList) != 0:
            # Right Arm
            angle = detector.findAngle(img, 12, 14, 16)
            # # Left Arm
            #angle = detector.findAngle(img, 11, 13, 15,False)
            per = np.interp(angle, (210, 310), (0, 100))
            bar = np.interp(angle, (220, 310), (650, 100))
            # print(angle, per)
     
            # Check for the dumbbell curls
            color = (255, 0, 255)
            if per == 100:
                color = (0, 255, 0)
                if dir == 0:
                    count += 0.5
                    dir = 1
            if per == 0:
                color = (0, 255, 0)
                if dir == 1:
                    count += 0.5
                    dir = 0
            print(count)
            
     
            # Draw Bar
            cv2.rectangle(img, (1100, 100), (1175, 650), color, 3)
            cv2.rectangle(img, (1100, int(bar)), (1175, 650), color, cv2.FILLED)
            cv2.putText(img, f'{int(per)} %', (1100, 75), cv2.FONT_HERSHEY_PLAIN, 4,
                        color, 4)
     
            # Draw Curl Count
            #cv2.rectangle(img, (0, 450), (250, 720), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, str(int(count)), (25, 670), cv2.FONT_HERSHEY_PLAIN, 10,
                        (255, 0, 0), 15)
     
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, 'FPS:'+str(int(fps)), (50, 100), cv2.FONT_HERSHEY_PLAIN, 2,
                    (255, 0, 0), 3)
     
       # cv2.imshow("Image", img)
        
        # Render Detections
       
        #cv2.imshow('MediaPipe feed', image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(img)
        cv2.waitKey(1)


if __name__ == '__main__':
    while run:
        reset = st.button('Stop')
        curlCOunter()
    while runT:
        reset = st.button('Stop')
        if(reset):
            count = 0
        startTrainer()
    else:
        st.write('Stopped')
        