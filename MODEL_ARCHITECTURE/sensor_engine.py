# coding: utf-8
import keras
import keras.backend as K
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten, Input, Lambda, Dropout
from keras.preprocessing.image import image, ImageDataGenerator
from keras.regularizers import l2


def sensorActivityRecEngine_CNN():
    
    model = Sequential()
    #model.add(Input(shape=(9, 98)))
    model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu",  input_shape=( 9, 98)))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Flatten())
 #   model.add(Dense(units=100,activation="relu"))
    model.add(Dense(units=2, activation="softmax"))
    model.summary()
    
    return model

def sensorActivityRecEngine_Logistic(NUM_ROWS,NUM_COLS):
   # this is a logistic regression in Keras

   # Build neural network
   model = Sequential()
   model.add(Dense(512, input_shape=(NUM_ROWS,NUM_COLS), activation='relu'))
   model.add(Dropout(0.5))
   model.add(Dense(256, activation='relu'))
   model.add(Dropout(0.25))
   model.add(Flatten())
   model.add(Dense(2, activation='softmax'))
   model.summary()
   return model
