# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 10:55:40 2018

@author: ratin
"""
import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
cnn_model = load_model("my_model.h5")

new_df = pd.read_csv("shibu_noyfirst.csv")
new_df.drop("Unnamed: 0",axis=1,inplace=True)
length = len(new_df)
train_len=int(length*0.8)
x_train = new_df.iloc[:train_len,1:].values/255
y_train = new_df.iloc[:train_len,0].values

x_test = new_df.iloc[train_len:,1:].values/255
y_test = new_df.iloc[train_len:,0].values

x_train, x_validate, y_train, y_validate = train_test_split(
    x_train, y_train, test_size=0.2, random_state=12345,
)

image = x_train[50, :].reshape((28, 28))
plt.imshow(image)
plt.show()

im_rows = 28
im_cols = 28
batch_size = 128
im_shape = (im_rows, im_cols, 1)

x_train = x_train.reshape(x_train.shape[0], *im_shape)
x_test = x_test.reshape(x_test.shape[0], *im_shape)
x_validate = x_validate.reshape(x_validate.shape[0], *im_shape)

print('x_train shape: {}'.format(x_train.shape))
print('x_test shape: {}'.format(x_test.shape))
print('x_validate shape: {}'.format(x_validate.shape))

cnn_model.fit(
        x_train, y_train, batch_size=batch_size,
        epochs=10, verbose=1,
        validation_data=(x_validate, y_validate),
    )

score = cnn_model.evaluate(x_test, y_test, verbose=0)

print('test loss: {:.4f}'.format(score[0]))
print(' test acc: {:.4f}'.format(score[1]))