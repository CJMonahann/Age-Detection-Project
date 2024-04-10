import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import random

'''
The random data that I created here is based off the idea of using the UTKFace dataset. This dataset has classifier information
for a face image's race, age, and gender.
'''
def main():
    face_names = ['man-face1', 'man-face2', 'woman-face1', 'woman-face2']
    data_classifiers = []
    new_size = 150
    features = []

    for i in range(50):
        image = 'face_data/' + random.choice(face_names) + '.jpg'
        img_arr = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        new_img = cv2.resize(img_arr, (new_size, new_size))
        new_img = np.array(new_img)
        features.append(new_img)
        data_classifiers.append(random.randint(0,40)) #append random age to be used for the example

    features = np.array(features).reshape(len(features), new_size, new_size, 1)
    features = features / 255.0

    data_classifiers = np.array(data_classifiers)

    #now feed through a neural network!
    model = Sequential()
    model.add(Conv2D(64, (3,3), input_shape = features.shape[1:]))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(64, (3,3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())
    model.add(Dense(64))

    model.add(Dense(1))
    model.add(Activation('relu'))

    model.compile(loss='mae',
                optimizer = 'adam',
                metrics=['accuracy'])

    model.fit(features, data_classifiers, batch_size = 10, epochs = 3, validation_split = 0.1)

    #now on to the prediction stage!
    pred = model.predict(features)

    #display results of testing the model
    num_person = 1
    for age in pred:
        format_age = format(age[0], '.2f')
        print(f'The Predicted age for Person {num_person} is {format_age} years old')
        num_person += 1

if __name__ == "__main__":
    main()