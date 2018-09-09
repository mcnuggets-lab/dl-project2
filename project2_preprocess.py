import os
import numpy as np
from skimage.io import imread, imshow
from matplotlib import pyplot as plt
#from skimage.transform import resize
import cv2
import pickle



# path constants
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
train_data = []
train_label = []
val_data = []
val_label = []
test_data = []

print("Reading train data...")
with open(TRAIN_LOC) as f:
    lines = f.readlines()
    train_data = np.empty((len(lines), *IMAGE_SIZE), dtype='float32')
    for ind, line in enumerate(lines):
        img_file, img_label = line.strip().split()
        img_file = os.path.join(DATA_FOLDER, img_file)
        raw_img = imread(img_file) / 255  # normalize
        img = cv2.resize(raw_img, (IMAGE_SIZE[1], IMAGE_SIZE[0]), interpolation = cv2.INTER_AREA)
        train_data[ind] = img
        label = [0] * 5
        label[int(img_label)] = 1
        train_label.append(label)
        
print("Read validation data...")
with open(VAL_LOC) as f:
    lines = f.readlines()
    val_data = np.empty((len(lines), *IMAGE_SIZE), dtype='float32')
    for ind, line in enumerate(lines):
        img_file, img_label = line.strip().split()
        img_file = os.path.join(DATA_FOLDER, img_file)
        raw_img = imread(img_file) / 255
        img = cv2.resize(raw_img, (IMAGE_SIZE[1], IMAGE_SIZE[0]), interpolation = cv2.INTER_AREA)
        val_data[ind] = img
        label = [0] * 5
        label[int(img_label)] = 1
        val_label.append(label)

print("Reading test data...")
with open(TEST_LOC) as f:
    lines = f.readlines()
    test_data = np.empty((len(lines), *IMAGE_SIZE), dtype='float32')
    for ind, line in enumerate(lines):
        img_file = line.strip()
        img_file = os.path.join(DATA_FOLDER, img_file)
        raw_img = imread(img_file) / 255
        img = cv2.resize(raw_img, (IMAGE_SIZE[1], IMAGE_SIZE[0]), interpolation = cv2.INTER_AREA)
        test_data[ind] = img
      


#train_data = np.array(train_data)
train_label = np.array(train_label)
#val_data = np.array(val_data)
val_label = np.array(val_label)
#test_data = np.array(test_data)


# sizes of the datasets
"""
print("Image size: {}".format(train_data[1].shape))  # variable size
print("Training size: {}".format(train_data.shape))  # 2569
print("Validation size: {}".format(val_data.shape))  # 550
print("Test size: {}".format(test_data.shape))       # 551
"""

# show a sample image
disp_ind = 1234
plt.imshow(train_data[disp_ind])

# save preprocessed data
data_dict = { "train_data": train_data,
              "train_label": train_label,
              "val_data": val_data,
              "val_label": val_label,
              "test_data": test_data }

pickle.dump(data_dict, open(PICKLE_PATH, 'wb'))
print("Preprocessed data saved as", PICKLE_PATH)



















