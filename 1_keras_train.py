#!./sensorEngineDevelopmentSetup/bin/python3
import math
import keras
import keras.backend as K
import tensorflow as tf
from keras.optimizers import SGD, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import to_categorical
import pandas

#from MODEL_ARCHITECTURE.vgg import return_VGG_ARCH
from MODEL_ARCHITECTURE.sensor_engine import sensorActivityRecEngine_CNN, sensorActivityRecEngine_Logistic
from MODULES.numpy_converter import process 


class sensorEngine_training_class():
	
	inputDataFileName=""
	trainedModelPath = "sensorNet.h5"

	# input image dimensions
	feature_rows, feature_cols = 9, 98

	def __init__(self,train_csv):
		self.inputDataFileName = train_csv

	def create_train_model(self):
		print ("\n Model creation started")
		model = sensorActivityRecEngine_Logistic(self.feature_rows, self.feature_cols)
		#sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
		opt = Adam(lr=0.0001)

		model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
		self.train_model(model)


	def train_model(self,model):
		#Training process
		print ("\n Training process started")
		checkpoint = ModelCheckpoint(self.trainedModelPath, monitor='acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
		early = EarlyStopping(monitor='acc', min_delta=0, patience=20, verbose=1, mode='auto')


		(X_featureData, Y_lblData) = process(self.inputDataFileName)
		Y_lblData = to_categorical(Y_lblData)

		print ('xlabel : ' + str(X_featureData.shape))
		print ('ylabel : ' + str(Y_lblData.shape))

		hist = model.fit( X_featureData, Y_lblData, epochs=30, batch_size=2, verbose=1, callbacks=[checkpoint,early] )

		'''
		train_datagen = ImageDataGenerator(
			rescale=1./255,
			shear_range=0.2,
			zoom_range=0.2,
			horizontal_flip=False)

		test_datagen = ImageDataGenerator(rescale=1./255)

		train_generator = train_datagen.flow_from_directory(
			'',
			target_size=(224, 224),
			batch_size=BATCH_SIZE,
			class_mode='categorical')

		print ("Found classes : ")
		label_map = (train_generator.class_indices)

		validation_generator = test_datagen.flow_from_directory(
			'/home/',
			target_size=(224, 224),
			batch_size=BATCH_SIZE,
			class_mode='categorical')


		hist = model.fit_generator(
		train_generator,
		steps_per_epoch= math.ceil(train_dataset_size/BATCH_SIZE),
		epochs=5,
		#validation_data=validation_generator,
		validation_steps=800,
		callbacks=[checkpoint,early])
		'''

	def convertModel_tflite(self):
		# Converting a tf.Keras model to a TensorFlow Lite model.
		print ("db1")
		converter = tf.lite.TFLiteConverter.from_keras_model(self.trainedModelPath)
		print ("db2")
		tflite_model = converter.convert()
		print ("db3")


	def __del__(self): 
		print('Destructor : remove training files') 
  

if __name__ == '__main__':

	trainFile = "train_2.csv"
	trainObj = sensorEngine_training_class(trainFile)

	trainObj.create_train_model()
	
	#Save mobile compatible tflite model
	trainObj.convertModel_tflite()