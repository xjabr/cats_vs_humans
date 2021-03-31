from prepare_dataset import get_data_set

import tensorflow as tf
from PIL import Image
import numpy as np
import cv2

def main():
    data, labels = get_data_set()

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(128, 128, 3)),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(50, activation=tf.nn.softmax)
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(data, labels, epochs=15)

    # prediction

    # load image
    img = cv2.imread('./dataset/test/human.jpeg', cv2.IMREAD_COLOR)
    img = cv2.resize(img, (128, 128))
    img_arr = tf.keras.preprocessing.image.img_to_array(img)
    img_arr = np.array([img_arr])

    predictions = model.predict(img_arr)
    print(np.argmax(predictions)) 

if __name__ == '__main__':
    main()