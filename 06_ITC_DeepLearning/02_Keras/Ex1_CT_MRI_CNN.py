import matplotlib.pyplot as plt
import numpy as np
import glob

from keras.models import Sequential, Model, Input
from keras import optimizers
from keras.layers import Dense, Activation, Conv2D, MaxPool2D, Flatten, BatchNormalization, Dropout
from keras.preprocessing.image import ImageDataGenerator


dataset_folder_path = 'MRI_CT_data'
train_folder = dataset_folder_path + '/train'
test_folder = dataset_folder_path + '/test'

test_files = glob.glob(test_folder + '/**/*.jpg')
train_files = glob.glob(train_folder + '/**/*.jpg')

#labels = ['mri' if tfile.find('mri') >= 0 else 'ct' for tfile in train_files]

train_examples = len(train_files)
test_examples = len(test_files)
batch_size = 32
print("Number of train examples: ", train_examples)
print("Number of test examples: ", test_examples)

"""View some sample images:"""

datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=5,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(
    rescale=1./255
)

img_height = img_width = 200
channels = 1
if channels == 1:
    color_mode_ = "grayscale"
else:
    color_mode_ = "rgb"

train_generator = datagen.flow_from_directory(
    train_folder,
    color_mode=color_mode_,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle=True,
    class_mode='binary'
)

validation_generator = test_datagen.flow_from_directory(
    test_folder,
    color_mode=color_mode_,
    target_size=(img_height, img_width),
    class_mode='binary'
)

"""## Convolution Neural Networks (CNN)"""

model = Sequential()

# TODO: Add a CNN:
# Note 1: The input_shape needs to be specified in this case (input_height, input_width, channels)
# Note 2: The order usually goes Conv2D, BatchNormalization, Activation, MaxPool,
# Note 3: Must be flattened before passing onto Dense layers
# Note 4: The loss is binary_crossentropy

# ------------------------------------------------------------------------------
# Training  
# ------------------------------------------------------------------------------

model.add(Conv2D(32, (3, 3), input_shape=(img_height, img_width, channels)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(1))
model.add(BatchNormalization())
model.add(Activation('sigmoid'))


adam = optimizers.adam()
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

model.fit_generator(train_generator,
                    validation_data=validation_generator,
                    epochs=100,
                    steps_per_epoch=train_examples//batch_size,
                    use_multiprocessing=True)


