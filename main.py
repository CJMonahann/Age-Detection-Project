import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import random, os, cv2

'''
Used to obtain directory and image file names within the utkcropped dataset directory.
Parameters: none
Returned: the absolute path of the dataset directory, array of .jpg image file names
'''
def collect_file_info():
    #constant filepath variables
    root_path = '.' #the root directory
    git_path = '.\\.git' #a path to a .git file
    i = 0

    for dir_path, dir_names, images in os.walk('.', topdown=False):
        if(dir_path != root_path and not dir_path.startswith(git_path)): #if the directory isn't related to the root or Git
            dir_path += "\\"
            return dir_path, images

'''
Uses the absolute path to the dataset, as well as .jpg names, to create both image feature data and classifiers
Parameters: absolute path to dataset directory, names of .jpg files within the directory
Returned: an array of the grayscaled image's feature data, image classifier data (age) 
'''    
def load_in_data(dir_path, file_arr, num_samples = 50):
    data_classifiers = []
    new_size = 150
    features = []

    #randomly select .jpg images to be used for the experiment (without replacement)
    img_names_arr = random.sample(file_arr, num_samples)

    for file_name in img_names_arr: #using the sub-sample of randomly selected .jpg image names
        img_path = dir_path + file_name #absolute path to an image
        split_name = file_name.split('_')
        img_age = int(split_name[0]) #age classifier taken frim the img name
        img_arr = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        new_img = cv2.resize(img_arr, (new_size, new_size))
        new_img = np.array(new_img)
        features.append(new_img)
        data_classifiers.append(img_age)

    #normalize features and re-addjust the np arrays
    features = np.array(features).reshape(len(features), new_size, new_size, 1)
    features = features / 255.0

    data_classifiers = np.array(data_classifiers)

    return features, data_classifiers

'''
Creates CNN model and returns model predicitons.
Parameters: image feature data, image classifier data, CNN batch size, CNN epochs, CNN validation split
Returned: a np array containing CNN model age predictions from the testing portion of the feature data (set by validation split) 
'''
def run_model(feature_data, classifier_data, batch_size = 10, epochs = 3, validation_split = 0.1):
    #now feed through a neural network!
    model = Sequential()
    model.add(Conv2D(64, (3,3), input_shape = feature_data.shape[1:]))
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

    model.fit(feature_data, classifier_data, batch_size = batch_size, epochs = epochs, validation_split = validation_split)

    #now on to the prediction stage!
    age_predictions = model.predict(feature_data)

    return age_predictions

'''
Displayes CNN model's predicted ages against the real ages of the used images. The absolute error for each predicition is also reported.
Parameters: np array of CNN model predicitons, classifier data from the images
Returned: none
'''
def display_results(model_predicitons, classifier_data):
    i = 0 #used to index classifier list and get the known age for each prediciton
    num_person = 1
    for age in model_predicitons:
        real_age = classifier_data[i]
        predicted_age = float(format(age[0]))
        prediciton_error = abs(real_age - predicted_age)
        prediciton_error = float(format(prediciton_error))
        print(f'Predicted age for Person {num_person}: {predicted_age} years old | Real Age - {real_age} | Error: {prediciton_error}')
        #print(f'Predicted age for Person {num_person}: {predicted_age} years old')
        num_person += 1
        i += 1

'''
main function that drives program logic at run-time
Parameters: none
Returned: none
'''
def main():
    #variables
    num_samples = 200

    #collect image directory information, load-in image data, and run CNN model
    dir_path, img_names = collect_file_info()
    feature_data, classifier_data = load_in_data(dir_path, img_names, num_samples)
    save_classifiers = classifier_data.copy()
    save_classifiers = [int(num) for num in save_classifiers]
    model_predicitons = run_model(feature_data, classifier_data)

    #display results of testing the model
    display_results(model_predicitons, save_classifiers)
    

if __name__ == "__main__":
    main()