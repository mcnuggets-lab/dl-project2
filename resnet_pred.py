import numpy as np
from keras.models import Model, load_model
from keras.utils import plot_model
import os, pickle

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



# load trained model
model = load_model("resnet18_ckpt.h5")

# print a summary of the model
model.summary()
plot_model(model, to_file='model.png')

# run prediction
preds = model.predict(test_data)
preds_class = np.argmax(preds, axis=1).T
np.savetxt("project2_01652721.txt", preds_class, fmt="%d")

print(preds_class)








