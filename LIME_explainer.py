import lime
import pandas as pd
import sklearn
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
from tensorflow.keras.models import load_model
from lime.wrappers import scikit_image
from functools import partial
import copy
from sklearn.linear_model import Ridge
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm

# best model
model = load_model('CNN_75.h5')


def limeExplainer(index, Xset, yset, num_features):
    test_image = Xset[index]
    # classification - probability
    preds = model.predict(Xset)
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    # show original image
    ax.imshow(test_image)
    if preds[index] > 0.5 and yset[index] == 1:
        print('Correctly Classified as Malignant!')
        print(preds[index])
    elif preds[index] < 0.5 and yset[index] == 0:
        print('Correctly Classified as Benign!!')
        print(preds[index])
    elif preds[index] > 0.5 and yset[index] == 0:
        print('Incorrect Classification! Model output is Malignant, but image is Benign.')
        print(preds[index])
    else:
        print('Incorrect Classification! Model output is Benign, but image is Malignant.')
        print(preds[index])

    # segmentation using quickshift algorithm
    KERNEL_SIZE, MAX_DIST, RATIO = 2, 10, 0.1  # variables
    segmentation_fn = scikit_image.SegmentationAlgorithm('quickshift', kernel_size=KERNEL_SIZE, max_dist=MAX_DIST, ratio=RATIO, random_seed=12)
    segments = segmentation_fn(test_image)

    # explanation generator
    explainer = lime_image.LimeImageExplainer()
    # background
    hide_color = 0
    fudged_image = test_image.copy()
    # If hide_color is not None, superpixels of hide_color used to create fudged_image, otherwise mean pixel color used
    if hide_color is None:
        for x in np.unique(segments):
            fudged_image[segments == x] = (
                np.mean(test_image[segments == x][:, 0]),
                np.mean(test_image[segments == x][:, 1]),
                np.mean(test_image[segments == x][:, 2]))
    else:
        fudged_image[:] = hide_color
    # Creating the perturbed samples and predicting their class y using the CNN model
    data, labels = explainer.data_labels(test_image, fudged_image, segments, classifier_fn=model.predict, num_samples=1000)


    def perturbed_images_generation(image, fudged_image, segments, data):
        rows = data
        imgs = []
        req_images = []
        # Iterate through each of the row of the array called data
        for row in rows:
            # temp a copy of the original image
            temp = copy.deepcopy(image)
            # Finding the indexes of the zeros in each row
            zeros = np.where(row == 0)[0]
            # create a temporary array called mask same shape as segments which contains all the values as False
            mask = np.zeros(segments.shape).astype(bool)
            # For each zero (off superpixel) in each row of data replace the image by the fudged_image
            for z in zeros:
                mask[segments == z] = True
            temp[mask] = fudged_image[mask]
            imgs.append(temp)
            req_images.extend(imgs)
            imgs = []
        return req_images

    # Perturbed images - can optionally show to make sure working correctly
    perturbed_images = perturbed_images_generation(test_image, fudged_image, segments, data)

    # Calculate the distances between perturbed images and original image
    distances = sklearn.metrics.pairwise_distances(data, data[0].reshape(1, -1), metric='cosine').ravel()

    # Find the top 3 closest perturbed images
    df = pd.DataFrame(distances, columns=['distance'])
    df1 = df.sort_values(by='distance')
    df1 = df1.drop_duplicates(keep='first')
    req_index = df1.index[1:4]

    # Exponential kernel
    def kernel(d, kernel_width):
        return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

    # exponential kernel with kernel width 25
    kernel_fn = partial(kernel, kernel_width=25)
    # Samples are given weights using exponential kernel
    weights = kernel_fn(distances)
    labels_column = labels[:, 0]

    # Ridge regression is fitted on the local data
    clf = Ridge(alpha=0.01, fit_intercept=True,)
    clf.fit(data, labels_column, sample_weight=weights)
    coef = clf.coef_

    num_features = num_features
    top_features = np.argsort(np.abs(coef))[-num_features:]
    print(top_features)
    print("All features: {}".format(len(coef)))

    # After getting the features Ridge regression is used to fit the local model
    model_regressor = Ridge(alpha=0.01, fit_intercept=True)
    easy_model = model_regressor
    easy_model.fit(data[:, top_features], labels_column, sample_weight=weights)
    prediction_score = easy_model.score(data[:, top_features], labels_column, sample_weight=weights)
    local_pred = easy_model.predict(data[0, top_features].reshape(1, -1))
    explainer = lime_image.LimeImageExplainer(feature_selection='forward_selection')

    # Defining the segmentation function
    segmentation_fn = SegmentationAlgorithm('quickshift', kernel_size=3, max_dist=20, ratio=0.25, random_seed=12)
    explanation = explainer.explain_instance(test_image, model.predict, top_labels=2, hide_color=0, num_samples=1000, segmentation_fn=segmentation_fn)
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, negative_only=False, num_features=num_features, hide_rest=False)

    # Code to show/save image explanation (turned off)
    # Turn on saving for running main.py if want to generate new images
    # Not needed for analysis

    # fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    # ax.imshow(mark_boundaries(temp, mask))
    # plt.savefig('lime-{}-{}.png'.format(index, num_features))
    # plt.show()

    # get pixel importance values for use in statistical comparisons
    pixel_values = mask
    mapped_values = []
    for i in range(pixel_values.shape[0]):
        row = i
        for j in range(pixel_values.shape[1]):
            col = j
            mapped_values.append((row, col, pixel_values[i][j]))
    dict_values = dict(((row, col), val) for row, col, val in mapped_values)

    return dict_values, len(coef)

