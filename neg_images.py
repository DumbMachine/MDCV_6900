# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 17:46:55 2018

@author: ratin
"""

# Face Recognition

import cv2
from PIL import Image
import numpy as np
import pandas as pd

# Loading the cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
#smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

# Defining a function that will do the detections
def detect(gray, frame):
    lis=[]
    lis_img=[]
    roi_color=0
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = gray[y:y+169, x:x+148]
        #cv2.imshow("rando",roi_color)
        lis=[x,x+w,y,y+h]
    return frame,lis,roi_color

video = cv2.VideoCapture(0)
ret = True
liss=[]
lis_img=[] 
roi_color=cv2.imread(r"C:\Users\ratin\Desktop\void.png")
temp_img=[0]
while ret == True:
    frame = video.read()[1] #_,frame = vid.read() causes problems sometimes
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    canvas,k,roi_color= detect(gray, frame)
    liss.append(k)
    if type(roi_color)!=int:
        temp_img = Image.fromarray(roi_color)
        temp_img = temp_img.resize((28,28),Image.BILINEAR)
        temp_img = np.array(temp_img)
        lis_img.append(temp_img)
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
'''lis=[0 for _ in range(len(lis_img))]
for i in range(len(lis_img)):
    lis[i]=lis_img[i].ravel()
#dictionary generation
data={}
#initsialize
for i in range(len(lis)):
    for j in range(784):#len(lis[0])
        data["pixel{}" .format(j)]=[]
        
for i in range(len(lis)):
    for j in range(784):#len(lis[0])       
        data["pixel{}" .format(j)].append(lis[i][j])
data["label"]=[]
for i in range(len(lis)):
    data["label"].append(1)
    
x = pd.DataFrame(data)
x.to_csv("auto.csv")'''




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
