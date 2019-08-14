import torch
from torchvision import transforms
import cv2
import numpy as np

import pyautogui

#Transform for incoming
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(size=256),
    transforms.CenterCrop(size=224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

#Load up the model
model = torch.load('resnet_model.pt', map_location='cpu')
model = model.module

#Dictionary of classes and the BGR value to display
classes = {0: ['looking', (35, 200, 45)], 1: ['not looking', (35, 45, 200)]}

#Webcam video capture
vid_capture = cv2.VideoCapture(0)
#vid_capture.set(cv2.CAP_PROP_FPS, 10)
# x_vid_res= int(vid_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
# y_vid_res = int(vid_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

#Bool for detecting if desktop is up or down
is_down = False

#Window to display video
cv2.namedWindow("Looking or not?")
cv2.startWindowThread()

while cv2.getWindowProperty('Looking or not?', 0) >= 0:
    #Read in video capture
    is_frame, frame = vid_capture.read()
    if not is_frame: break
    
    #Convert image from BGR to RGB then transform
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = transform(img)

    #Get prediction from the model
    hypothesis = model(img.unsqueeze(0))
    _, pred = torch.max(hypothesis, 1)

    #Dislplay text and image
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(frame, classes[int(pred)][0], (0,100), font, 3, classes[int(pred)][1], 3, cv2.LINE_AA)
    cv2.imshow('Looking or not?', frame)

    #Example of application of looking/not-looking
    if(int(pred) == 1 and is_down is False):
        is_down= True
        pyautogui.keyDown('winleft')
        pyautogui.press('d')
        pyautogui.keyUp('winleft')
    if(int(pred) == 0 and is_down is True):
        is_down= False
        pyautogui.keyDown('winleft')
        pyautogui.press('d')
        pyautogui.keyUp('winleft')

    #Press escape key to exit
    key = cv2.waitKey(50)
    if key == 27:
        break
    else:
        print(key)
        
vid_capture.release()
cv2.destroyAllWindows()