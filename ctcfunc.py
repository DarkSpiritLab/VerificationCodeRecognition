from captcha.image import ImageCaptcha
import matplotlib.pyplot as plt
import numpy as np
import random

import string
characters = string.digits + string.ascii_uppercase
print(characters)

width, height, n_len, n_class = 170, 80, 4, len(characters)+1

from keras import backend as K

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

#define model
from keras.models import *
from keras.layers import *
rnn_size = 128

input_tensor = Input((width, height, 3))
x = input_tensor
for i in range(3):
    x = Convolution2D(32, (3, 3), activation='relu')(x)
    x = Convolution2D(32, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

conv_shape = x.get_shape()
x = Reshape(target_shape=(int(conv_shape[1]), int(conv_shape[2]*conv_shape[3])))(x)

x = Dense(32, activation='relu')(x)

gru_1 = GRU(rnn_size, return_sequences=True, init='he_normal', name='gru1')(x)
gru_1b = GRU(rnn_size, return_sequences=True, go_backwards=True, init='he_normal', name='gru1_b')(x)
gru1_merged = merge([gru_1, gru_1b], mode='sum')

gru_2 = GRU(rnn_size, return_sequences=True, init='he_normal', name='gru2')(gru1_merged)
gru_2b = GRU(rnn_size, return_sequences=True, go_backwards=True, init='he_normal', name='gru2_b')(gru1_merged)
x = merge([gru_2, gru_2b], mode='concat')
x = Dropout(0.25)(x)
x = Dense(n_class, init='he_normal', activation='softmax')(x)
base_model = Model(input=input_tensor, output=x)

labels = Input(name='the_labels', shape=[n_len], dtype='float32')
input_length = Input(name='input_length', shape=[1], dtype='int64')
label_length = Input(name='label_length', shape=[1], dtype='int64')
loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([x, labels, input_length, label_length])

model = Model(input=[input_tensor, labels, input_length, label_length], output=[loss_out])
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adadelta')


#load model
model.load_weights('model.h5')

import cv2
characters2 = characters + ' '
import base64
def getCode(imgo):
    print(type(imgo))
    imgo=base64.b64decode(bytes(imgo,'utf-8'))
    print(type(imgo))
    with open('tempimg','wb+') as f:
        f.write(imgo)
        f.flush()

    img = cv2.imread('tempimg')
    # cv2.imshow("img",img)
    # cv2.waitKey(0)
    img = cv2.resize(img, (170, 80), interpolation = cv2.INTER_CUBIC)
    nimg = np.rollaxis(img, 0, 2)
    # print(nimg.shape)
    x_sou = np.broadcast_to(nimg, (1, 170, 80, 3))
    print(x_sou.shape)
    # cv2.imshow("image",img)
    # cv2.waitKey(0)
    y_pred = base_model.predict(x_sou)
    y_pred = y_pred[:, 2:, :]
    out = K.get_value(K.ctc_decode(y_pred, input_length = np.ones(y_pred.shape[0]) * y_pred.shape[1], )[0][0])[:, :4]
    out = ''.join([characters[x] for x in out[0]])
    # print(x_pre)
    argmax = np.argmax(y_pred, axis = 2)[0]
    print(argmax)
    str = ""
    str = ''.join([characters2[x] for x in argmax])
    str = str.replace(' ', '')
    print(str)
    return str
    # list(zip(argmax, ''.join([characters2[x] for x in argmax])))