# coding: utf-8
import keras
import keras.backend as K
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten, Input, Lambda
from keras.preprocessing.image import image, ImageDataGenerator
from keras.regularizers import l2


def sensorActivityRecEngine_CNN( rows, cols, depth, mClasses):
    
    model = Sequential()
    model.add(Conv2D(input_shape=(rows, cols, depth),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Flatten())
 #   model.add(Dense(units=100,activation="relu"))
    model.add(Dense(units=mClasses, activation="softmax"))
    model.summary()
    
    return model