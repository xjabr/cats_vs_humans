import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

BATCH_WIDTH = 28
BATCH_HEIGHT = 28
LABEL_NAME_BY_VAL = [ 'Human', 'Cat' ]

def get_dataset_train():
    x_data = []
    y_data = []

    files_face = glob.glob('./dataset/train/faces/*.jpg')
    files_cats = glob.glob('./dataset/train/cats/*.jpg')

    j = 0
    while j < 500:
        for i in range(5):
            img = cv2.imread(files_face[i + j], cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (BATCH_WIDTH, BATCH_HEIGHT))
            x_data.append(img)
            y_data.append(0)  # 0 = humans

        for i in range(5):
            img = cv2.imread(files_cats[i + j], cv2.IMREAD_COLOR)  # load image
            # convert rgb image to scale gray
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (BATCH_WIDTH, BATCH_HEIGHT))  # resize image
            x_data.append(img)
            y_data.append(1)  # 1 = cats

        j += 5

    print('Batch size ', len(x_data))

    # normalizate dataset
    x_data = tf.keras.utils.normalize(x_data, axis=1)

    # convert images array to np.array
    x_data = np.array(x_data)
    y_data = np.array(y_data)

    return x_data, y_data


def get_dataset_test():
    """
    Generate the dataset for predictions and valutate the model
    """
    x_data = []
    y_data = []

    files_face = glob.glob('./dataset/test/faces/*.jpg')
    files_cats = glob.glob('./dataset/test/cats/*.jpg')

    j = 0
    while j < 500:
        for i in range(5):
            img = cv2.imread(files_face[i + j], cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (BATCH_WIDTH, BATCH_HEIGHT))
            x_data.append(img)
            y_data.append(0)  # 0 = humans

        for i in range(5):
            img = cv2.imread(files_cats[i + j], cv2.IMREAD_COLOR)  # load image
            # convert rgb image to scale gray
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (BATCH_WIDTH, BATCH_HEIGHT))  # resize image
            x_data.append(img)
            y_data.append(1)  # 1 = cats

        j += 5

    print('Batch size ', len(x_data))

    x_data = tf.keras.utils.normalize(x_data, axis=1)

    x_data = np.array(x_data)
    y_data = np.array(y_data)

    return x_data, y_data
