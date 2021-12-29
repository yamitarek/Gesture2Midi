
# IMPORTS
import mediapipe as mp
import cv2
import numpy as np
import uuid
import os
import time
import matplotlib.pyplot as plt
from scipy import fftpack
from oscpy.client import OSCClient

# setup osc connection

OSC_HOST ="127.0.0.1" #127.0.0.1 is for same computer
OSC_PORT = 8000
OSC_CLIENT = OSCClient(OSC_HOST, OSC_PORT)

# media pipe objects
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# get labels for hands
def get_label(index, hand, results):
    output = None
    for idx, classification in enumerate(results.multi_handedness):
        if classification.classification[0].index == index:
            
            # Process results
            label = classification.classification[0].label
            score = classification.classification[0].score
            text = '{} {}'.format(label, round(score, 2))
            
            # Extract Coordinates
            coords = tuple(np.multiply(
                np.array((hand.landmark[mp_hands.HandLandmark.WRIST].x, hand.landmark[mp_hands.HandLandmark.WRIST].y)),
            [cam_width,cam_height]).astype(int))
            
            output = text, coords
            
    return output


# draw finger position
def draw_finger_position(image, results, joint_list):

    #BUFFER Variable
    buff = np.array([0,0])
    
    # Loop through hands
    for hand in results.multi_hand_landmarks:
        #Loop through joint sets 
        for joint in joint_list:
            a = np.array([hand.landmark[joint[0]].x, hand.landmark[joint[0]].y]) # First coord
            b = np.array([hand.landmark[joint[1]].x, hand.landmark[joint[1]].y]) # Second coord
            c = np.array([hand.landmark[joint[2]].x, hand.landmark[joint[2]].y]) # Third coord
            
            txt_a = str(round(a[0], 2)) + ", " + str(round(a[1], 2))
                
            cv2.putText(image, txt_a, tuple(np.multiply(a, [cam_width, cam_height]).astype(int)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            

            
            hand_ind = results.multi_hand_landmarks.index(hand)
            joint_ind = joint_list.index(joint)

            string_path = '/'+'h'+str(hand_ind)+'/'+'f'+str(joint_ind)+'/x'
            ruta = string_path.encode()
            if (buff[0] != a[0]):
                OSC_CLIENT.send_message(ruta, [float(a[0])])
            string_path = '/'+'h'+str(hand_ind)+'/'+'f'+str(joint_ind)+'/y'
            ruta = string_path.encode()
            if (buff[1] != a[1]):
                OSC_CLIENT.send_message(ruta, [float(a[1])])
            
            buff = a

               
    return image


# joint List 
joint_list = [[8,7,6], [12,11,10], [16,15,14], [20,19,18]]

# Video Capture 
## this is where you choose your webcam. try 0, 1, etc. 
cap = cv2.VideoCapture(1)
# camera parameters
cam_width  = cap.get(3)  # float `width`
cam_height = cap.get(4)  # float `height`

#camera hardcoded for camo / iphone parameters
#cam_width = 960
#cam_height = 540

# camera parameters
print(cam_width, " ", cam_height)

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands: 
    while cap.isOpened():
        ret, frame = cap.read()
        
        # BGR 2 RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Flip on horizontal
        image = cv2.flip(image, 1)
        
        # Set flag
        image.flags.writeable = False
        
        # Detections
        results = hands.process(image)
        
        # Set flag to true
        image.flags.writeable = True
        
        # RGB 2 BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Detections
        #print(results)
        
        # Rendering results
        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=(250, 0, 250), thickness=2, circle_radius=2),
                                         )
                
                # Render left or right detection
                if get_label(num, hand, results):
                    text, coord = get_label(num, hand, results)
                    cv2.putText(image, text, coord, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Draw angles to image from joint list
            #draw_finger_angles(image, results, joint_list)

            # Draw position to image from joint list
            draw_finger_position(image, results, joint_list)
            
        # Save our image    
        #cv2.imwrite(os.path.join('Output Images', '{}.jpg'.format(uuid.uuid1())), image)
        cv2.imshow('Hand Tracking', image)

        # Quit application by pressing 'q' key
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)