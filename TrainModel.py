import tensorflow as tf
import os
import random, math
import numpy as np

from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
import cv2, joblib

X_data = []
i = 0
print("Importing X_data......")
for i in range(18000):
    img_path = './data/newdata/img/File ' + str(i) + '.jpg'
    img = cv2.imread(img_path,1)
    img = resize(img, (128, 128))
    X_data.append(img)
X_data = np.array(X_data)
    

Y_mask = []
i = 0
print("Importing Y_mask......")
for i in range(18000):
    img_path = './data/newdata/mask/File ' + str(i) + '.png'
    img = cv2.imread(img_path,0)
    img = resize(img, (128, 128))
    Y_mask.append(img)
Y_mask = np.array(Y_mask)

# Binarise Mask Data
maskShape = Y_mask.shape
for i in range(maskShape[0]):
        for j in range(maskShape[1]):
            for k in range(maskShape[2]):
                if Y_mask[i,j,k] > 0.5:
                    Y_mask[i,j,k] = 1
                if Y_mask[i,j,k] <= 0.5:
                    Y_mask[i,j,k] = 0

# Load Model
print("Training Model....")
model = tf.keras.models.load_model('./final/UNet-1024-30k')
checkpointer = tf.keras.callbacks.ModelCheckpoint('model_for_nuclei.h5', verbose=1, save_best_only=True)
callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir='logs')]
results = model.fit(X_data, Y_mask, validation_split=0.1, batch_size=16, epochs=20, callbacks=callbacks)

print("Saving Model....")
model.save('./final/UNet-1024-48k')
joblib.dump(results, './final/UNet-1024-48k-hist')


# Testing the Model
model = tf.keras.models.load_model('./final/UNet-1024-1k')
X_test = []
print("Importing X_test......")
i = 18001
for i in range(37):
    img_path = './data/newdata/img/File ' + str(i) + '.jpg'
    img = cv2.imread(img_path,1)
    img = resize(img, (128, 128))
    X_test.append(img)
X_test = np.array(X_test)
    
Y_test = []
print("Importing Y_test......")
i = 18001
for i in range(37):
    img_path = './data/newdata/mask/File ' + str(i) + '.png'
    img = cv2.imread(img_path,0)
    img = resize(img, (128, 128))
    Y_test.append(img)
Y_test = np.array(Y_test)

# Binarise Mask Data
print("Binarising the mask.....")
maskShape = Y_test.shape
for i in range(maskShape[0]):
        for j in range(maskShape[1]):
            for k in range(maskShape[2]):
                if Y_test[i,j,k] > 0.5:
                    Y_test[i,j,k] = 1
                if Y_test[i,j,k] <= 0.5:
                    Y_test[i,j,k] = 0

print("Evaluating Testing Accuracy...")
results = model.evaluate(X_test, Y_test)

