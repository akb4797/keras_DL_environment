import math
import keras
from keras.optimizers import SGD, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping

#from MODEL_ARCHITECTURE.vgg import return_VGG_ARCH
from MODEL_ARCHITECTURE.sensor_engine import sensorActivityRecEngine


# Design model
img_channels = 1
nb_classes = 3
# input image dimensions
img_rows, img_cols = 6, 2000

model = sensorActivityRecEngine_CNN(rows=img_rows, cols=img_cols, depth=img_channels, mClasses=nb_classes)
#sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
opt = Adam(lr=0.0001)

model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])


checkpoint = ModelCheckpoint("sensorNet.h5", monitor='acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='acc', min_delta=0, patience=20, verbose=1, mode='auto')

#Training process
train_dataset_size = 187400

BATCH_SIZE = 32

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
    '/home/ubuntu/3_dataSpace/SR_v27/',
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