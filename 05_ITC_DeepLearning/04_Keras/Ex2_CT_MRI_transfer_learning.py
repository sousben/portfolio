# Load util
import matplotlib.pyplot as plt

import numpy as np
import glob

from keras.models import Sequential, Model
from keras import optimizers
from keras.layers import Dense, Activation, Conv2D, MaxPool2D, Flatten, BatchNormalization, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras import applications
from keras.applications import VGG16
from keras import models
from keras import layers
from keras import optimizers


img_height = img_width = 150
channels = 3
if (channels == 1):
    color_mode_ = "grayscale"
else:
    color_mode_ = "rgb"


dataset_folder_path = 'MRI_CT_data'
train_folder = dataset_folder_path + '/train'
test_folder = dataset_folder_path + '/test'

test_files = glob.glob(test_folder + '/**/*.jpg')
train_files = glob.glob(train_folder + '/**/*.jpg')

train_examples = len(train_files)
test_examples = len(test_files)
print("Number of train examples: " , train_examples)
print("Number of test examples: ", test_examples)

# ---------------------------------------------------------------------


#Load the VGG model
vgg_conv = VGG16(include_top=False, input_shape=(img_height, img_width, channels))
vgg_conv.summary(line_length=150)

# Freeze some layers
for layer in vgg_conv.layers[:-4]:
    layer.trainable = False

# Check the trainable status of the individual layers
for layer in vgg_conv.layers:
    print(layer, layer.trainable, layer.output_shape)

# Create the model
model = models.Sequential()

# Add the vgg convolutional base model
model.add(vgg_conv)

# Add new layers
model.add(Flatten())
model.add(Dense(8192))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(2))
model.add(BatchNormalization())
model.add(Activation('softmax'))

for layer in model.layers:
    print(layer, layer.trainable)

# Show a summary of the model. Check the number of trainable parameters
model.summary()

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1. / 255)

# Change the batchsize according to your system RAM
train_batchsize = 100
val_batchsize = 10

train_generator = train_datagen.flow_from_directory(
    train_folder,
    target_size=(img_height, img_width),
    batch_size=train_batchsize,
    class_mode='categorical')


# Compile the model
adam = optimizers.adam(lr=0.0001)
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])


validation_generator = validation_datagen.flow_from_directory(
    test_folder,
    target_size=(img_height, img_width),
    batch_size=val_batchsize,
    class_mode='categorical',
    shuffle=False) 

# Train the model
history = model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.samples / train_generator.batch_size,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples / validation_generator.batch_size,
    verbose=1,
    use_multiprocessing=True
)

# ---------------------------------------------------------------------
# test your accuracy
# ---------------------------------------------------------------------

# YOUR CODE
print('Testing accuracy...')
score = model.evaluate_generator(validation_generator,
                                 validation_generator.samples/validation_generator.batch_size,
                                 workers=12,
                                 verbose=True)

print("Loss: %.2f Accuracy: %.2f%%" % (score[0], score[1]*100))
