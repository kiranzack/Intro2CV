import os
import tensorflow as tensor
from tensorflow import keras
import numpy as np
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten, Dropout
from tqdm import tqdm
from keras.layers import Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

#Storing paths of all the images available in the dataset to a list. 
fullImagePath = []
for root, dirs, files in os.walk(".", topdown=False): 
    for name in files:
        path = os.path.join(root, name)
        if path.endswith("JPG"):  # only .JPG file paths are retrieved and stored in the list
            fullImagePath.append(path)

#Reading all the images converting them from RGB to GRAY scale, resizing them and storing them into a numpy array. 
#Labels corresponding to each image are extracted through the image path. 
imageData = [] 
label = []
for path in tqdm(fullImagePath): #tqdm is used to display a progress bar just to keep a check on the number of iterations. 
    path = str(path)
    im = cv2.imread(path) 
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) 
    im = cv2.resize(im, (120, 120))
    imageData.append(im)
    
    l= path.split("\\")[2] #Label extraction from image path of format ./Dataset/0/Image0.JPG
    l= int(l)
    label.append(l)

imageData = np.array(imageData, dtype= np.uint8) #forming numpy array for images and their corresponding labels 
label = np.array(label, dtype= np.uint8)

imageData = imageData.reshape(-1, 120,120, 1)
imageData = imageData.astype('float32')
imageData = imageData / 255.0

# splitting the training data into train and validation sets. train set - 70% of the total dataset 
img_train, img_test, l_train, l_test = train_test_split(imageData, label, test_size=0.30, random_state=13)

# Building a Convolutional Neural Network Model using LeakyReLU activation function. 
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',padding='same',input_shape=(120,120,1)))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D((2, 2),padding='same'))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
model.add(LeakyReLU(alpha=0.1))                  
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(128, activation='linear'))
model.add(LeakyReLU(alpha=0.1))           
model.add(Dropout(0.3))
model.add(Dense(52, activation='softmax'))

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(img_train, l_train, epochs=20, batch_size=64, verbose=2, validation_data=(img_test, l_test)) # fitting the model to the training data
model.save('mudras_model_2.h5') #saving the model 