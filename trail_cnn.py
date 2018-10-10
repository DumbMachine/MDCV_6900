import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.optimizers import Adam
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from PIL import Image
import cv2

batch_size = 128
num_classes = 2
epochs = 30


df = pd.read_csv("auto7000.csv")
df.drop("Unnamed: 0",axis=1,inplace=True)
train_7000 = df.iloc[:-1500,:]
test_7000 = df.iloc[-1500:,:]
train_df = pd.read_csv("fashion-mnist_train.csv")
test_df=pd.read_csv("fashion-mnist_test.csv")
train =[train_7000,train_df]
test = [test_7000,test_df]
train_data = pd.concat(train)
test_data = pd.concat(test)

x_train = train_data.iloc[:, 1:].values / 255
y_train = train_data.iloc[:, 0].values

x_test = test_data.iloc[:, 1:].values / 255
y_test = test_data.iloc[:, 0].values


x_train, x_validate, y_train, y_validate = train_test_split(
    x_train, y_train, test_size=0.2, random_state=12345,
)

image = x_train[50, :].reshape((28, 28))

plt.imshow(image)
plt.show()


im_rows = 28
im_cols = 28
batch_size = 214
im_shape = (im_rows, im_cols, 1)

x_train = x_train.reshape(x_train.shape[0], *im_shape)
x_test = x_test.reshape(x_test.shape[0], *im_shape)
x_validate = x_validate.reshape(x_validate.shape[0], *im_shape)

print('x_train shape: {}'.format(x_train.shape))
print('x_test shape: {}'.format(x_test.shape))
print('x_validate shape: {}'.format(x_validate.shape))

cnn_model = Sequential([
    Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=im_shape),
    MaxPooling2D(pool_size=2),
    Dropout(0.2),
    
    Flatten(),
    Dense(32, activation='relu'),
    Dense(11, activation='softmax')
])

cnn_model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(lr=0.001),
    metrics=['accuracy']
)
cnn_model.fit(
    x_train, y_train, batch_size=batch_size,
    epochs=10, verbose=1,
    validation_data=(x_validate, y_validate),
)

score = cnn_model.evaluate(x_test, y_test, verbose=0)

print('test loss: {:.4f}'.format(score[0]))
print(' test acc: {:.4f}'.format(score[1]))

#Saving The Model
from keras.models import load_model

cnn_model.save("my_model.h5")
del cnn_model

cnn_model = load_model("my_model.h5")

#predicting from a value
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def detect(gray, frame):
    roi_color=0
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        if type(roi_gray)!=int:
            roi_gray = Image.fromarray(roi_gray)
            roi_gray = roi_gray.resize((28,28),Image.BILINEAR)
            roi_gray = np.array(roi_gray)
        cv2.putText(frame,"{}".format(name),(x,y),cv2.FONT_HERSHEY_SIMPLEX , 0.5,(255,255,255),2,cv2.LINE_AA)
        #cv2.imshow("rando",roi_color)
        lis=[x,x+w,y,y+h]
    return frame,roi_color

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
video.release()
cv2.destroyAllWindows()

