from dataset import get_dataset_train, get_dataset_test

import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

def main():
    data, labels = get_dataset_train()

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(data, labels, epochs=25)

    # generation loss and acc graph time
    plt.title('loss')
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.legend()
    plt.show()

    # load dataset for test
    test_data, test_labels = get_dataset_test()
    predictions = model.predict(test_data)
    
    corrects = 0
    wrongs = 0
    for i in range(len(predictions)):
        val_predicted = np.argmax(predictions[i])
        if val_predicted == test_labels[i]:
            corrects += 1
        else:
            wrongs += 1

    print('Corrects %d\nWrongs %d' % (corrects, wrongs))

    plt.bar('Corrects', corrects, width=0.3, color='g')
    plt.bar('Wrongs', wrongs, width=0.3, color='r')
    plt.ylabel('Value')
    plt.show()


    answer = input("Do you want save the model? [y][n] ")
    if answer == 'Y' or answer == 'y':
        model.save('model/')

if __name__ == '__main__':
    main()