#!./sensorEngineDevelopmentSetup/bin/python3
import os
import sys
import time
import numpy as np
import tensorflow as tf
from tensorflow import lite
from MODULES.numpy_converter import process, getGT

#Inference class prototype in python
#Interface can be ported to C/C++/Java
class tfLiteInferenceInterface():
	model_file = ''
	
	def __init__(self,modelInp):
		self.model_file = modelInp

	def predictActivity(self, inpStream):
		#Do inferencing on inptut data
		#print ("Inferencing input 9x98 ..")
		interpreter = tf.lite.Interpreter(model_path=self.model_file)
		interpreter.allocate_tensors()

		input_details = interpreter.get_input_details()
		output_details = interpreter.get_output_details()

		# check the type of the input tensor
		floating_model = input_details[0]['dtype'] == np.float32
		# NxHxWxC, H:1, W:2
		height = input_details[0]['shape'][1]
		width = input_details[0]['shape'][2]

		# add N dim
		input_data = np.expand_dims(inpStream, axis=0)

		if floating_model:
			input_data = (np.float32(input_data) - 0) / 1

		interpreter.set_tensor(input_details[0]['index'], input_data)

		start_time = time.time()
		interpreter.invoke()
		stop_time = time.time()

		output_data = interpreter.get_tensor(output_details[0]['index'])
		results = np.squeeze(output_data)
		#print('Inference time : {:.3f}ms'.format((stop_time - start_time) * 1000))

		return results

	def __del__(self):
		pass

def main():
	modelFile = 'sensorNet_lite.tflite'
	inputDataFileName = 'test_2.csv'
	labels = ['0','1','2','3','4','5','6','7']


	tfLite_inferObj = tfLiteInferenceInterface(modelFile)

	(X_featureData, Y_lblData) = process(inputDataFileName)

	ctr = 0 
	for elem in X_featureData:
		results = tfLite_inferObj.predictActivity(elem)

		top_k = results.argsort()[-5:][::-1]
		print ('Predicted Class : '  + str(top_k[0]) + ' -> GT : '+ str(Y_lblData[ctr]).strip())
		print ()
		ctr += 1
		#print('{}: {:08.6f}'.format(top_k[0], float(results[i])))
		# for i in top_k:
		# 	print('{:08.6f}: {}'.format(float(results[i]), labels[i]))

if __name__=='__main__':
	main()


