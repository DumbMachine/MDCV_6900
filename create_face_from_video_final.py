import numpy as np
import cv2
from imutils.video import FileVideoStream
import sys
import imutils
from imutils.video import FPS,FileVideoStream
import pandas as pd
from PIL import Image
from scipy import ndimage
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
def detect(gray, frame):
    roi_gray=0
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        
        #cv2.putText(frame,"{}".format("rat"),(x,y),cv2.FONT_HERSHEY_SIMPLEX , 0.5,(255,255,255),2,cv2.LINE_AA)
        #cv2.imshow("rando",roi_color)
    return frame,roi_gray
lis_img=[]
frames=0
cap = cv2.VideoCapture(r"C:\Users\ratin\Downloads\videoplayback.mp4")
fps = FPS().start()
cap.set(3,100)
cap.set(4,50)
while(cap.isOpened()):
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=450)
    if ret==True:
        frames+=1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        frame,roi_gray = detect(gray,gray)
        if type(roi_gray)!=int:
                roi_gray = Image.fromarray(roi_gray)
                roi_gray = roi_gray.resize((28,28),Image.BILINEAR)
                roi_gray = np.array(roi_gray)
                lis_img.append(roi_gray)
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        cap.release()
        cv2.destroyAllWindows()
fps.stop()
print("[INFO] fps is: {}".format(frames/fps.elapsed()))
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
cap.release()
cv2.destroyAllWindows()

lis=[0 for _ in range(len(lis_img))]
for i in range(len(lis_img)):
    lis[i]=lis_img[i].ravel()
#dictionary generation
data={}
#initsializeasd
data["label"]=[]
for i in range(len(lis)):
    data["label"].append(69)
for i in range(len(lis)):
    for j in range(784):#len(lis[0])
        data["pixel{}" .format(j+1)]=[]
        
for i in range(len(lis)):
    for j in range(784):#len(lis[0])       
        data["pixel{}" .format(j+1)].append(lis[i][j])

    
x = pd.DataFrame(data)
x.to_csv("fromvideotrial.csv")

def create_face_from_video(filename):
    frames=0
    lis_img=[]
    fvs = FileVideoStream(filename).start()
    time.sleep(1.0)
    fps = FPS().start()
    
    while fvs.more():
        frame = fvs.read()
        frame = imutils.resize(frame, width=450)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = np.dstack([frame, frame, frame])
        if ret==True:
            frames+=1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #gray = ndimage.rotate(gray,90)
            frame,roi_gray = detect(gray,gray)
            if type(roi_gray)!=int:
                    roi_gray = Image.fromarray(roi_gray)
                    roi_gray = roi_gray.resize((28,28),Image.BILINEAR)
                    roi_gray = np.array(roi_gray)
                    lis_img.append(roi_gray)
            cv2.imshow('frame',frame)
            fps.update()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            cap.release()
            cv2.destroyAllWindows()
        
    fps.stop()
    cap.release()
    cv2.destroyAllWindows()
    print("hehe: ",frames/fps.elapsed())
    lis=[0 for _ in range(len(lis_img))]
    for i in range(len(lis_img)):
        lis[i]=lis_img[i].ravel()
    #dictionary generation
    data={}
    #initsializeasd
    data["label"]=[]
    for i in range(len(lis)):
        data["label"].append(69)
    for i in range(len(lis)):
        for j in range(784):#len(lis[0])
            data["pixel{}" .format(j+1)]=[]
            
    for i in range(len(lis)):
        for j in range(784):#len(lis[0])       
            data["pixel{}" .format(j+1)].append(lis[i][j])
    
        
    x = pd.DataFrame(data)
    x.to_csv("shibu.csv")

create_face_from_video("WIN_20181012_13_27_57_Pro.mp4")
    
