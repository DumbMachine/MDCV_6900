#---------Using Video Feed------------------------------------------------------
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def detect(gray,frame):
    faces = face_cascade.detectMultiScale(gray,1.3,5)#frame,scaletoreduce2,minneighbours
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+h,y+h),(255,0,0),2)
    return frame
        
video= cv2.VideoCapture(0)
while(1):
    check, frame = video.read()
    status=0
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    canvas = detect(gray,frame)
    cv2.imshow("Canvas",canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
#--------------------------------------------------------------------------------
