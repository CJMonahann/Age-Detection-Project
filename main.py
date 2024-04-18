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
def run_model(feature_data, feature_classifiers, test_data, batch_size = 10, epochs = 3, validation_split = 0.1):
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

    model.compile(loss='mae', optimizer = 'adam')

    model.fit(feature_data, feature_classifiers, batch_size = batch_size, epochs = epochs, validation_split = validation_split)

    #now on to the prediction stage!
    age_predictions = model.predict(test_data)

    return age_predictions

'''
Displayes CNN model's predicted ages against the real ages of the used images. The absolute error for each predicition is also reported.
Parameters: np array of CNN model predicitons, classifier data from the images
Returned: none
'''
def get_results(model_predicitons, classifier_data):
    i = 0 #used to index classifier list and get the known age for each prediciton
    num_person = 1
    avg_error = 0
    for age in model_predicitons:
        real_age = classifier_data[i]
        predicted_age = float(format(age[0]))
        prediciton_error = abs(real_age - predicted_age)
        avg_error += prediciton_error
        num_person += 1
        i += 1
    avg_error = round(avg_error / i) #this will give us the mean absolute error
    return avg_error

'''
Restores the classifier data from its np array to that of a traditional array within integers
Parameters: a np array
Returned: an array of type int
'''
def restore_classifiers(arr):
    real_nums = arr.copy()
    real_nums = [int(num) for num in real_nums]
    return real_nums

'''
Part of the 'all_trials' function, this fucntion collects feature and testing data for a trial, and
calls for the running of the CNN model.
Parameters: directory path, arr of image names, number of samples to test on CNN, CNN batch_size,
CNN epochs, CNN validation split.
Returned: the average absolute error derived from the CNN model for the specific trial run
'''
def execute_trial(dir_path, img_names, num_samples, batch_size, epochs, validation_split):
    feature_data, feature_classifiers = load_in_data(dir_path, img_names, num_samples) #feature data to test the model
    test_data, test_classifiers = load_in_data(dir_path, img_names, num_samples) #new random sample to test with
    predicition_classifiers = restore_classifiers(test_classifiers) #save test classifier data in original format to be used when calculating error
    model_predicitons = run_model(feature_data, feature_classifiers, test_data, batch_size, epochs, validation_split)
    #display results of testing the model
    trials_data = get_results(model_predicitons, predicition_classifiers)
    return trials_data

'''
Part of the experimentation, this function runs the CNN model for each of the specified sample sizes provided,
as well as configures the CNN model for the specific experiment taking place.
Parameters: directory path, arr of image names, arr of configured sample sizes to test, dict of configurations
for the current experiment taking place.
Returned: an arr with three average absolute error values recorded for the three trials.
'''
def all_trials(dir_path, img_names, sample_sizes, curr_config):
    trials_data = []

    #extract needed configurations
    batch_size = curr_config["batch_size"]
    epochs = curr_config["epochs"]
    validation_split = curr_config["validation_split"]

    for num_samples in sample_sizes: #represents the three trials per configuarion
        avg_error = execute_trial(dir_path, img_names, num_samples, batch_size, epochs, validation_split)
        trials_data.append(avg_error)
    return trials_data

'''
Runs the three different experiments, each with their own configuarions, for a total of three times. 
Per experiment, each of the three trials is done using a differen't number of sample sizes, as specified by the configured number or samples.
Parameters: directory path, arr of image names, arr of configured sample sizes to test, dicitonary with 
all experiment configurations for the three different tests.
Returned: a dictionary containint the info about, as well as data, for all three experiments.
'''
def run_experiment(dir_path, img_names, sample_sizes, experiment_configs):
    MAX_TEST = 3
    num_test = 1

    all_experiments = {}

    while num_test <= MAX_TEST:
        #get current experiment configs
        key = f'test_{num_test}'
        curr_config = experiment_configs[key]
        trials_data = all_trials(dir_path, img_names, sample_sizes, curr_config) #run all three trials for the current test configurations

        #add relevant experiment results to 'temp_dict' dicitonary
        temp_dict = {}
        temp_key = f'config_{num_test}'
        temp_dict[temp_key] = curr_config
        temp_dict["sample_sizes"] = sample_sizes
        temp_dict["results"] = trials_data

        #add the 'temp_dict' containing all relevant experiment information to the 'all_experiments' dictionary
        all_experiments[key] = temp_dict
        num_test += 1

    #return data that can be used to construct graphs and/or tables
    return all_experiments

'''
main function that drives program logic at run-time
Parameters: none
Returned: none
'''
def main():
    #set values to be tested
    experiment_configs = {
        "test_1": {
            "batch_size": 10,
            "epochs": 3,
            "validation_split": 0.1
        },
        "test_2": {
            "batch_size": 30,
            "epochs": 4,
            "validation_split": 0.3
        },
        "test_3": {
            "batch_size": 50,
            "epochs": 5,
            "validation_split": 0.5
        }
    }
    sample_sizes = [100,200,300]

    #collect image directory information, load-in image data, and run CNN model
    dir_path, img_names = collect_file_info()

    #return data from all experiments that can be used to graph and/or make tables
    experiments_data = run_experiment(dir_path, img_names, sample_sizes, experiment_configs)
 
 
if __name__ == "__main__":
    main()