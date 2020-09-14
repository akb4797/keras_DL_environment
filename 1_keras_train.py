import math
import keras
from keras.optimizers import SGD, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping

#from MODEL_ARCHITECTURE.vgg import return_VGG_ARCH
from MODEL_ARCHITECTURE.resnet import ResnetBuilder

BATCH_SIZE = 32

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

stateList = []
for x, y in label_map.items():
  print(x, y) 
  z = str(str(x) + ',' + str(y))
  stateList.append(z)

def sortTrackID(elem):
	return int(elem.split(',')[1])

stateList.sort(key=sortTrackID)
classLabelsFile = open('classLbl.txt','w+')
for elem in stateList:
    classLabelsFile.write( elem.split(',')[0]+'\n')
print (stateList)
classLabelsFile.close()

validation_generator = test_datagen.flow_from_directory(
    '/home/ubuntu/3_dataSpace/SR_v27/',
    target_size=(224, 224),
    batch_size=BATCH_SIZE,
    class_mode='categorical')


# Design model
'''
model = return_VGG_ARCH()
model.load_weights( 'vgg16_1.h5', by_name=True)
'''
img_channels = 3
nb_classes = 42
# input image dimensions
img_rows, img_cols = 224,224

model = ResnetBuilder.build_resnet_50((img_channels, img_rows, img_cols), nb_classes)
#sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
opt = Adam(lr=0.0001)

model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])


checkpoint = ModelCheckpoint("build_resnet_50.h5", monitor='acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='acc', min_delta=0, patience=20, verbose=1, mode='auto')

#Training process
train_dataset_size = 187400


hist = model.fit_generator(
train_generator,
steps_per_epoch= math.ceil(train_dataset_size/BATCH_SIZE),
epochs=5,
#validation_data=validation_generator,
validation_steps=800,
callbacks=[checkpoint,early])