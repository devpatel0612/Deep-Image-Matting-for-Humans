import tensorflow as tf
import os, skimage
import random, math
import numpy as np

from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
import cv2, joblib


# Testing the Model
model = tf.keras.models.load_model('./final/UNet-1024-48k')
X_test = []
print("Importing X_test......")
i = 18001
for i in range(37):
    img_path = './data/newdata/img/File ' + str(i) + '.jpg'
    img = skimage.io.imread(img_path,1)
    img = resize(img, (128, 128, 3))
    X_test.append(img)
X_test = np.array(X_test)

Y_test = []
print("Importing Y_test......")
i = 18001
for i in range(37):
    img_path = './data/newdata/mask/File ' + str(i) + '.png'
    img = skimage.io.imread(img_path,0)
    img = resize(img, (128, 128))
    Y_test.append(img)
Y_test = np.array(Y_test)

print("Predicting the output......")
X_mask = model.predict(X_test)[:,:,:,0]

k = 16
plt.figure()
plt.imshow(X_test[k])
plt.show()
plt.figure()
plt.imshow(X_mask[k], cmap="gray")
plt.show()
plt.figure()
plt.imshow(Y_test[k], cmap="gray")
plt.show()

Output = X_test*0
M,N = X_mask[k].shape
for i in range(M):
    for j in range(N):
        if (X_mask[k][i][j]>0.4):
            Output[k][i][j][:]= X_test[k][i][j][:]
        else:
            Output[k][i][j][:] = 1

plt.figure()
plt.imshow(Output[k])
plt.show()    


