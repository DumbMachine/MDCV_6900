# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 22:52:39 2018

@author: ratin
"""

# Face Recognition
'''since in this technique most of the faces are similar for the particular person can we use one datapoint as 2 and double the dataset for a person'''
import cv2
from PIL import Image
import numpy as np
import pandas as pd
from keras.models import load_model

cnn_model = load_model("my_model.h5")

# Loading the cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
#smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

# Defining a function that will do the detections
def detect(gray, frame):
    roi_gray=[0]
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        if type(roi_gray)!=int:
            roi_gray = Image.fromarray(roi_gray)
            roi_gray = roi_gray.resize((28,28),Image.BILINEAR)
            roi_gray = np.array(roi_gray)
            roi_gray=roi_gray.reshape([1,28,28,1])
            pred = cnn_model.predict(roi_gray)
            pred=pred.reshape(-1,1)
            for i in range(len(pred)):
                if pred[i]==max(pred):
                    k=i
                    break
            if k==10:#add nos of pictures captured and time eta and time remaining.
                name="mom"
            elif k==11:
                name="rat"
            #elif k==11:
                #name="Mami"
            else:
                name="nothing"
            cv2.putText(frame,name,(x,y),cv2.FONT_HERSHEY_SIMPLEX , 0.5,(255,255,255),2,cv2.LINE_AA)

    return frame,roi_gray

video = cv2.VideoCapture(0)
ret = True
temp_img=[0]
while ret == True:
    frame = video.read()[1] #_,frame = vid.read() causes problems sometimes
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    canvas,roi_color= detect(gray, frame)
    cv2.imshow('Video', canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# =============================================================================
# last_frame= Image.fromarray(last_frame)
# last_frame= last_frame.resize((28,28),Image.BILINEAR)
# last_frame= np.array(last_frame)
# cv2.imwrite("last_image_28x28.jpg",last_frame)
# =============================================================================
video.release()
cv2.destroyAllWindows()


    





#w=50-60
#h=150-160

# =============================================================================
# from PIL import Image
# img = frame
# img = img.resize((28,28), Image.BILINEAR)
# pil_img = Image.fromarray(img)
# pil_img = np.array(pil_img)
# =============================================================================


#CHecking the captured image
# =============================================================================
# for i in range(len(lis_img)):
#     if i==len(lis_img)-1:
#         cv2.destroyAllWindows()
#     else:
#         cv2.imshow("rando",np.array(lis_img[i]))
#         cv2.waitKey(100)
# =============================================================================

# =============================================================================
# 
# for i in lis_img:
#     data[1]=i.ravel()
# =============================================================================
# # =============================================================================
# data={}
# for i in range(len(temp_ravel)):
#     data["pixel {}" .format(j)]=int(temp_ravel.reshape(-1,1)[i])
#     j+=1
# x = pd.DataFrame(data,index=[0]) when we want to put single array
# x = pd.DataFrame(data)
# =============================================================================
# =============================================================================
# 
# lis=[0 for _ in range(len(lis_img))]
# for i in range(len(lis_img)):
#     lis[i]=lis_img[i].ravel()
#     
#     
# for i in range(len(lis)):
#     for j in range(784):#len(lis[0])
#         data["pixel{}" .format(j)] = lis[i][j]
#         
#         
#         
# for i in range(len(lis)):
#     for j in range(784):#len(lis[0])
#         data["pixel{}" .format(j)]=[]
#         
# 
# for i in range(len(lis)):
#     for j in range(784):#len(lis[0])
#         data["pixel{}" .format(j)].append(lis[i][j])
#     
# =============================================================================
