#!./sensorEngineDevelopmentSetup/bin/python3
import os
import sys
import subprocess
import math
import numpy as np
import cv2

import matplotlib.pyplot as plt

from keras.layers import Dense, Conv2D, MaxPool2D , Flatten, Input, Lambda
from keras.models import Sequential, Model
import keras.backend as K
from keras.preprocessing.image import image, ImageDataGenerator
from keras.regularizers import l2

from sklearn.metrics.pairwise import cosine_similarity
from keras.utils import to_categorical


from MODULES.numpy_converter import process, getGT

from MODEL_ARCHITECTURE.sensor_engine import sensorActivityRecEngine_Logistic


class sensorEngine_inference_class():
    inputDataFileName = ""
    trainedModelName = 'sensorNet.h5'
    
    feature_rows, feature_cols = 9, 98
    gt_labels = np.empty(100)

    def __init__(self,testFile):
        self.inputDataFileName = testFile 

    def get_GT(self):
        return getGT(self.inputDataFileName)


    def get_classifierOutput(self):
        saved_model = sensorActivityRecEngine_Logistic(self.feature_rows, self.feature_cols)
        saved_model.load_weights( self.trainedModelName)

        (X_featureData, Y_lblData) = process(self.inputDataFileName)

        if (X_featureData.any()):
            Y_lblData = to_categorical(Y_lblData)
            self.gt_labels = Y_lblData

            output = saved_model.predict([X_featureData])

            return output
    
        else:
            return np.empty(1)





def main():

    test_file = "test_2.csv"
    workout_ID = ['0','1','2','3','4','5','6','7']

    inferObj = sensorEngine_inference_class(test_file)
    gtLabels = inferObj.get_GT()
    outputClass = inferObj.get_classifierOutput()
    
    if(outputClass.any()):
        ctr = 0
        for elem in outputClass:
            print ("GroundTruth WorkOutID: " + str(gtLabels[ctr][0])+ " \t -> \t Predicted WorkOutID : " + str(workout_ID[int( elem.argmax(axis=-1))] + ' \t Confidence ' + str(max(elem))))
            ctr += 1


    
if __name__ == "__main__":
    main()
    
