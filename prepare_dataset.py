import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt

def get_data_set():
    x_data = []
    y_data = []

    files_face = glob.glob('./dataset/train/faces/*.jpg')
    files_cats = glob.glob('./dataset/train/cats/*.jpg')

    for item in files_face:
        img = cv2.imread(item, cv2.IMREAD_COLOR)
        # resize img to 32x32px
        img = cv2.resize(img, (128, 128))
        x_data.append(img)
        y_data.append(0) # 0 = humans
    
    for item in files_cats:
        img = cv2.imread(item, cv2.IMREAD_COLOR)
        # resize img to 32x32px
        img = cv2.resize(img, (128, 128))
        x_data.append(img)
        y_data.append(1) # 1 = cats

    # convert images array to np.array
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    print(x_data.shape) # (900, 128, 128, 3)

    return x_data, y_data