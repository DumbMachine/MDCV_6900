import cv2

# Loading the cascades
face_cascade = cv2.CascadeClassifier('myfacedetector.xml')
#smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

# Defining a function that will do the detections
def detect(gray, frame):
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
    return frame

video = cv2.VideoCapture(0)
ret = True
while ret == True:
    frame = video.read()[1] #_,frame = vid.read() causes problems sometimes
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    canvas = detect(gray, frame)
    cv2.imshow('Video', canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video.release()
cv2.destroyAllWindows()