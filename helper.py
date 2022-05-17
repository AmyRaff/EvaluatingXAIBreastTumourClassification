from os import listdir
import cv2
import imutils
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

# This file contains helper functions for loading and cropping data.


# Function to load data from files to arrays of images (X) and labels (y)
def load_data(dir_list, image_size):
    X = []
    y = []
    image_width, image_height = image_size
    for directory in dir_list:
        for filename in listdir(directory):
            image = cv2.imread(directory + '\\' + filename)  # open
            image = crop_brain_contour(image, plot=False)  # crop
            image = cv2.resize(image, dsize=(image_width, image_height), interpolation=cv2.INTER_CUBIC)  # resize
            image = image / 255.  # normalize
            X.append(image)
            if directory[-9:] == 'Malignant': # generate labels
                y.append([1])
            else:
                y.append([0])
    X = np.array(X)
    y = np.array(y)
    print(f'Number of examples is: {len(X)}')
    print(f'X shape is: {X.shape}')
    print(f'y shape is: {y.shape}')
    return X, y


# Function for data preprocessing, taken from the original brain-tumour-detection github we got the CNN from
# https://github.com/MohamedAliHabib/Brain-Tumor-Detection
# removed image blurring and randomization, otherwise the same
def crop_brain_contour(image, plot=False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])
    new_image = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]
    new_image = new_image[10:217, 10:217]
    if plot:
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.tick_params(axis='both', which='both',
                        top=False, bottom=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)
        plt.title('Original Image')
        plt.subplot(1, 2, 2)
        plt.imshow(new_image)
        plt.tick_params(axis='both', which='both',
                        top=False, bottom=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)
        plt.title('Cropped Image')
        plt.show()
    return new_image


# Function to calculate F1 score of a model, used for performance evaluation
def compute_f1_score(y_true, preds):
    preds = np.asarray(preds)
    y_pred = np.where(preds > 0.5, 1, 0)
    score = f1_score(y_true, y_pred)
    return score


# this file initialises our data and splits it into training, validation and testing sets
# this is done in this file to ensure that it only needs to be done once

datapath = 'data/All/'
malignant_data = datapath + 'Malignant'
benign_data = datapath + 'Benign'
IMG_WIDTH, IMG_HEIGHT = (227, 227)
X, y = load_data([malignant_data, benign_data], (IMG_WIDTH, IMG_HEIGHT))


# remove randomization with random seed
def split_data(X, y,test_size=0.05):
    X_train, X_test_val, y_train, y_test_val = train_test_split(X, y, test_size=test_size, random_state=9)
    X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, test_size=0.5, random_state=9)

    return X_train, y_train, X_val, y_val, X_test, y_test


# final data for use in experiments
X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y, test_size=0.05)


