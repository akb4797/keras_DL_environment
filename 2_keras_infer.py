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

from MODEL_ARCHITECTURE.vgg import return_VGG_ARCH, return_VGG_ARCH_FE

BATCH_SIZE = 32

img_height = 224
img_width = img_height

DATA_FOLDER = ''

classLabelsFile = open('classLbl.txt','r')
tempList = classLabelsFile.readlines()
stateList = []
for elem in tempList:
    stateList.append(elem.strip())

print ("Supported classes : " +  str(stateList))


def get_classifierOutput(refPath):

    input_images = [] # Store resized versions of the images here.

    saved_model = return_VGG_ARCH( returnLayer='softmax')
    saved_model.load_weights( 'vgg16_0.h5')
    #print ("\nLoading with Weights : vgg16_1.h5")

    refimg = image.load_img(refPath, target_size=(img_height, img_width))
    img = image.img_to_array(refimg) 
    input_images.append(img)
    input_images = np.array(input_images)


    output = saved_model.predict([input_images])
    return output
    

def initialize_weights(shape, name=None):
    return np.random.normal(loc = 0.0, scale = 1e-2, size = shape)



def initialize_bias(shape, name=None):
    return np.random.normal(loc = 0.5, scale = 1e-2, size = shape)



def get_siamese_model(m_inpWidth, m_inpHeight, m_channels):
    """
        Siamese Model architecture
    """
    # Define the tensors for the two input images
    left_input = Input( name='ref_inp', shape=(m_inpWidth, m_inpHeight, m_channels))
    right_input = Input( name='query_inp', shape=(m_inpWidth, m_inpHeight, m_channels))

    #Trained backBone model
    saved_model = return_VGG_ARCH_FE()
    saved_model.load_weights( 'vgg16_1.h5', by_name=True)
    # Generate the encodings (feature vectors) for the two images
    encoded_l = saved_model(left_input)
    encoded_r = saved_model(right_input)
    L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]), name='L1Distance')
    L1_distance = L1_layer([encoded_l, encoded_r])
    prediction = Dense(1,activation='sigmoid',  kernel_initializer=initialize_weights,bias_initializer=initialize_bias)(L1_distance)
    siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)
    #siamese_net.summary()


    # Debug output
    debug_model = saved_model
    DEBUG_LAYER_NAME = 'featureVector'
    intermediate_layer_model = Model(inputs=debug_model.get_input_at(0),
                                    outputs=debug_model.get_layer(DEBUG_LAYER_NAME).output)
    
    db_encoded_l = intermediate_layer_model(left_input)
    db_encoded_r = intermediate_layer_model(right_input)
    db_L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]), name='L1Distance')
    db_L1_distance = db_L1_layer([db_encoded_l, db_encoded_r])
    prediction = Dense(1,activation='relu',  kernel_initializer=initialize_weights,bias_initializer=initialize_bias)(db_L1_distance)
    db_siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)

 
    return db_siamese_net, saved_model


def main():

    test_path = os.path.join(DATA_FOLDER, 'IN_t3', '1.png')

    outputClass = get_classifierOutput(test_path)
    print ("\tClass : " + str(outputClass))

    print ("\tProbable State : " + str(stateList[int( outputClass.argmax(axis=-1))]))

    '''
    for subFldr in os.listdir(DATA_FOLDER):
        print ('\nState : ' + subFldr)
        
        ref_path = os.path.join(DATA_FOLDER, 'FL_t1', '1.png')
        
        for img in os.listdir(os.path.join(DATA_FOLDER, subFldr)):
            if(1):
                query_path = os.path.join(DATA_FOLDER, subFldr, img)
                # print ("\tRef img : " + ref_path)
                # print ("\tQuery img : " + query_path )
                
                outputClass = get_classifierOutput(query_path)
                outputSimilarity, fe_simCosine = get_imageSimilarity(ref_path, query_path)

                if(fe_simCosine >= 0.99):
                    print("\tMatching reference design")
                    print ("\tProbable State : " + str(stateList[int( outputClass.argmax(axis=-1))]))
                    print("\tSimilarity Output : " + str(fe_simCosine))

    '''
if __name__ == "__main__":
    main()
    
