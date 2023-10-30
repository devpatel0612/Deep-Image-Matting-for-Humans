import tensorflow as tf
import os
import random
import numpy as np

from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
import cv2, joblib

# X_data = []
# print("Importing X_data......")
# for i in range(1000):
#     img_path = './data/img/' + str(i) + '.jpg'
#     img = cv2.imread(img_path,1)
#     img = resize(img, (128, 128))
#     X_data.append(img)
# X_data = np.array(X_data)
    

# Y_mask = []
# print("Importing Y_mask......")
# for i in range(1000):
#     img_path = './data/mask/' + str(i) + '.png'
#     img = cv2.imread(img_path,0)
#     img = resize(img, (128, 128))
#     Y_mask.append(img)
# Y_mask = np.array(Y_mask)

# # Binarise Mask Data
# maskShape = Y_mask.shape
# for i in range(maskShape[0]):
#         for j in range(maskShape[1]):
#             for k in range(maskShape[2]):
#                 if Y_mask[i,j,k] > 0.5:
#                     Y_mask[i,j,k] = 1
#                 if Y_mask[i,j,k] <= 0.5:
#                     Y_mask[i,j,k] = 0

# Define Height, width and channels of the model   
IMG_HEIGHT = 128
IMG_WIDTH = 128
IMG_CHANNELS = 3

#Build the model
inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

#Contraction path
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
c1 = tf.keras.layers.Dropout(0.1)(c1)
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
c2 = tf.keras.layers.Dropout(0.1)(c2)
c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
 
c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
c3 = tf.keras.layers.Dropout(0.2)(c3)
c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)
 
c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
c4 = tf.keras.layers.Dropout(0.2)(c4)
c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)
 
c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
c5 = tf.keras.layers.Dropout(0.2)(c5)
c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
p5 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c5)
 
c6 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p5)
c6 = tf.keras.layers.Dropout(0.2)(c6)
c6 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
p6 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c6)

c7 = tf.keras.layers.Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p6)
c7 = tf.keras.layers.Dropout(0.3)(c7)
c7 = tf.keras.layers.Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

#Expansive path 
u8 = tf.keras.layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c7)
u8 = tf.keras.layers.concatenate([u8, c6])
c8 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
c8 = tf.keras.layers.Dropout(0.2)(c8)
c8 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
 
u9 = tf.keras.layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c8)
u9 = tf.keras.layers.concatenate([u9, c5])
c9 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
c9 = tf.keras.layers.Dropout(0.2)(c9)
c9 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

u10 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c9)
u10 = tf.keras.layers.concatenate([u10, c4])
c10 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u10)
c10 = tf.keras.layers.Dropout(0.2)(c10)
c10 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c10)
 
u11 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c10)
u11 = tf.keras.layers.concatenate([u11, c3])
c11 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u11)
c11 = tf.keras.layers.Dropout(0.2)(c11)
c11 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c11)
 
u12 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c11)
u12= tf.keras.layers.concatenate([u12, c2])
c12 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u12)
c12 = tf.keras.layers.Dropout(0.1)(c12)
c12 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c12)
 
u13 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c12)
u13 = tf.keras.layers.concatenate([u13, c1], axis=3)
c13 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u13)
c13 = tf.keras.layers.Dropout(0.1)(c13)
c13 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c13)
 
outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c13)
 
model = tf.keras.Model(inputs=[inputs], outputs=[outputs])

opt = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics='accuracy')
model.summary()

# ################################
# #Modelcheckpoint
# checkpointer = tf.keras.callbacks.ModelCheckpoint('model_for_nuclei.h5', verbose=1, save_best_only=True)
# callbacks = [
#         tf.keras.callbacks.TensorBoard(log_dir='logs')]

# results = model.fit(X_data, Y_mask, validation_split=0.1, batch_size=16, epochs=20, callbacks=callbacks)

# model.save('./final/UNet-1024-1k')
# joblib.dump(results, './final/UNet-1024-1k-hist')

# # Testing the Model
# X_test = []
# print("Importing X_test......")
# i = 34000
# for i in range(424):
#     img_path = './data/img/' + str(i) + '.jpg'
#     img = cv2.imread(img_path,1)
#     img = resize(img, (128, 128))
#     X_test.append(img)
# X_test = np.array(X_data)
    
# Y_test = []
# print("Importing Y_test......")
# i = 34000
# for i in range(424):
#     img_path = './data/mask/' + str(i) + '.png'
#     img = cv2.imread(img_path,0)
#     img = resize(img, (128, 128))
#     Y_test.append(img)
# Y_test = np.array(Y_mask)

# # Binarise Mask Data
# maskShape = Y_test.shape
# for i in range(maskShape[0]):
#         for j in range(maskShape[1]):
#             for k in range(maskShape[2]):
#                 if Y_test[i,j,k] > 0.5:
#                     Y_test[i,j,k] = 1
#                 if Y_test[i,j,k] <= 0.5:
#                     Y_test[i,j,k] = 0

# results = model.evaluate(X_test, Y_test)

