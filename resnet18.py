import numpy as np
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, Dropout
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.initializers import glorot_uniform
from keras.callbacks import ModelCheckpoint, EarlyStopping
import os, pickle

from blocks import identity_block_straight, convolutional_block_straight

import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)



# define constants and read in data
DATA_FOLDER = os.path.join(".", "data")
TRAIN_LOC = os.path.join(DATA_FOLDER, "train.txt")
VAL_LOC = os.path.join(DATA_FOLDER, "val.txt")
TEST_LOC = os.path.join(DATA_FOLDER, "test.txt")

# other constants
IMAGE_SIZE = (128, 128, 3)
CLASS = {0: "daisy", 1: "dandelion", 2: "roses", 3: "sunflowers", 4: "tulips"}
NUM_LABELS = len(CLASS)

# path to save preprocessed file
PICKLE_PATH = os.path.join(DATA_FOLDER,
             "data_" + str(IMAGE_SIZE[0]) + "x" + str(IMAGE_SIZE[1]) + ".pickle")

# read in train images, and resize all of them to the same size
data_dict = pickle.load(open(PICKLE_PATH, "rb"))
train_data = data_dict["train_data"]
train_label = data_dict["train_label"]
val_data = data_dict["val_data"]
val_label = data_dict["val_label"]
test_data = data_dict["test_data"]

# model hyperparameters
NUM_EPOCHES = 100
BATCH_SIZE = 64


def resnet18(input_shape=IMAGE_SIZE, classes=5):
    """
    Implementation of the ResNet18 architecture.

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """

    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)

    # Stage 1
    X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block_straight(X, f=3, filters=[64, 64], stage=2, block='a', s=1)
    X = identity_block_straight(X, 3, [64, 64], stage=2, block='b')

    # Stage 3
    X = convolutional_block_straight(X, f=3, filters=[128, 128], stage=3, block='a', s=2)
    X = identity_block_straight(X, 3, [128, 128], stage=3, block='b')

    # Stage 4
    X = convolutional_block_straight(X, f=3, filters=[256, 256], stage=4, block='a', s=2)
    X = identity_block_straight(X, 3, [256, 256], stage=4, block='b')

    # Stage 5
    X = convolutional_block_straight(X, f=3, filters=[512, 512], stage=5, block='a', s=2)
    X = identity_block_straight(X, 3, [512, 512], stage=5, block='b')

    # final pooling
    X = AveragePooling2D(pool_size=(2, 2), name='avg_pool')(X)

    # output layer
    X = Flatten()(X)
    X = Dense(512, activation='relu', name='fch1', kernel_initializer=glorot_uniform(seed=0))(X)
    X = Dropout(0.5)(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)

    # Create model
    model = Model(inputs=X_input, outputs=X, name='ResNet18')

    return model


# Running the model
adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model = resnet18(input_shape = IMAGE_SIZE, classes = 5)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

# resume training if necessary
#model = load_model("resnet18_ckpt.h5")



# image augmentation before each batch
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

datagen.fit(train_data)

# enforce early stopping and checkpointing
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')
checkpoint = ModelCheckpoint("resnet18_ckpt.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)

# Model fitting
history = model.fit_generator(datagen.flow(train_data, train_label, batch_size = BATCH_SIZE),
                              steps_per_epoch=len(train_data) / BATCH_SIZE, epochs = NUM_EPOCHES, validation_data=(val_data, val_label),
                              callbacks=[early, checkpoint])

# Train model with all data we have
#train_data_prod = np.vstack((train_data, val_data))
#train_label_prod = np.vstack((train_label, val_label))
#history = model.fit_generator(datagen.flow(train_data_prod, train_label_prod, batch_size = BATCH_SIZE),
#                              steps_per_epoch=len(train_data) / BATCH_SIZE, epochs = NUM_EPOCHES)

# validation
preds = model.evaluate(val_data, val_label)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))

# show loss history
loss_history = history.history["loss"]
print(loss_history)

# save model
model.save("resnet18_50e.h5")


