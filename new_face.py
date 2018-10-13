import cv2
import numpy as np
from PIL import Image
import pandas as pd
import os.path
from imutils.video import FPS

# Loading the cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')
# Defining a function that will do the detections
def detect(gray, frame):
    roi_gray=0
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        
        #cv2.putText(frame,"{}".format("rat"),(x,y),cv2.FONT_HERSHEY_SIMPLEX , 0.5,(255,255,255),2,cv2.LINE_AA)
        #cv2.imshow("rando",roi_color)
    return frame,roi_gray
def create_face():
    name=str(input("Enter the name of the Person: "))
    
    try :
        dic=np.load("real_dic.npy").item()
        for i in dic:
            if dic[i]==name:
                print("This Name is Already there ")
                return "Thenga"
        dic[max(dic)+1]=name
    except FileNotFoundError:
        dic={}
        dic[10]=name
    
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10,50)
    fontScale              = 1
    fontColor              = (255,255,255)
    lineType               = 2
    frames=0
    frames_add=0
    lis_img=[]
    video = cv2.VideoCapture(0)
    fps=FPS().start()
    ret = True
    while ret == True:
        frame = video.read()[1] #_,frame = vid.read() causes problems sometimes
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        canvas,roi_gray = detect(gray, frame)
        if type(roi_gray)!=int:
            roi_gray = Image.fromarray(roi_gray)
            roi_gray = roi_gray.resize((28,28),Image.BILINEAR)
            roi_gray = np.array(roi_gray)
            lis_img.append(roi_gray)
            frames_add+=1
        cv2.putText(canvas,"Frames Added: {}".format(frames_add),
                    bottomLeftCornerOfText, 
                    font, 
                    fontScale,
                    fontColor,
                    lineType)
        cv2.imshow('Video', canvas)
        frames+=1
        if frames_add==5500:
            break
        fps.update()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            video.release()
            break
    fps.stop()
    print("fps : {}".format(frames/fps.elapsed()))
    print("Total frames {}".format(frames))
    print("Total time(in seconds): {}".format(fps.elapsed()))
    print("Frames added are: {}".format(frames_add))
    print("Frame percentage: {}".format(-frames/frames_add))
    video.release()
    cv2.destroyAllWindows()
    lis=[0 for _ in range(len(lis_img))]
    for i in range(len(lis_img)):
        lis[i]=lis_img[i].ravel()
    #dictionary generation
    data={}
    #initsialize
    data["label"]=[]
    for i in range(len(lis)):
        data["label"].append(max(dic)+1)
    for i in range(len(lis)):
        for j in range(784):#len(lis[0])
            data["pixel{}" .format(j+1)]=[]
            
    for i in range(len(lis)):
        for j in range(784):#len(lis[0])       
            data["pixel{}" .format(j+1)].append(lis[i][j])
    
        
    x = pd.DataFrame(data)
    print("Succesfully added {} frames.".format(len(x)))
    choice=int(input("Enter one to save: "))
    if choice==1:
        print("saving")
        if os.path.isfile(name+"1.csv"):
            print("file with this names exists.EXITING!!!")
            return
        x.to_csv(name+"1.csv")
        print("saved as {}.csv".format(name))
        print("has key: {}".format(max(dic)+1))
        np.save("real_dic",dic)
        return
    
create_face()
    
