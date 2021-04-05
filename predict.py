import tensorflow as tf
import numpy as np
import cv2

def main(path, path_model):
  img = cv2.imread(path, cv2.IMREAD_COLOR)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  img = cv2.resize(img, (300, 300))
  img_arr = tf.keras.preprocessing.image.img_to_array(img)
  img_arr = np.array([img_arr])

  model = tf.keras.models.load_model(path_model)

  predictions = model.predict(img_arr)
  print(np.argmax(predictions)) 

if __name__ == '__main__':
  main('./dataset/test/dog.jpeg', './model')