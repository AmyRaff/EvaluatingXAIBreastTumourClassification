import matplotlib.pyplot as plt
from keras.preprocessing import image
import pylab as pl
from skimage.segmentation import slic
from tensorflow.keras.models import load_model
import shap
import numpy as np
import shap.explainers
from matplotlib.colors import LinearSegmentedColormap

# best model
model = load_model('CNN_75.h5')


def shapExplainer(index, Xset, yset):
    img_orig = Xset[index]
    img = image.array_to_img(img_orig)
    n_segs = 100
    # segment the image so we don't have to explain every pixel
    segments_slic = slic(img, n_segments=n_segs, compactness=20, sigma=1)

    # define a function that depends on a binary mask representing if an image region is hidden
    def mask_image(zs, segmentation, image, background=None):
        if background is None:
            background = image.mean((0, 1))
        out = np.zeros((zs.shape[0], image.shape[0], image.shape[1], image.shape[2]))
        for i in range(zs.shape[0]):
            out[i,:,:,:] = image
            for j in range(zs.shape[1]):
                if zs[i, j] == 0:
                    out[i][segmentation == j, :] = background
        return out

    # generate predictions
    def f(z):
        return model.predict(mask_image(z, segments_slic, img_orig, 0))

    preds = model.predict(Xset)

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

    # use Kernel SHAP to explain the network's predictions
    explainer = shap.KernelExplainer(f, np.zeros((1,n_segs)))
    shap_values = explainer.shap_values(np.ones((1,n_segs)), nsamples=1000)

    # make a color map
    colors = []
    for l in np.linspace(1,0,100):
        colors.append((245/255,39/255,87/255,l))
    for l in np.linspace(0,1,100):
        colors.append((24/255,196/255,93/255,l))
    cm = LinearSegmentedColormap.from_list("shap", colors)

    def fill_segmentation(values, segmentation):
        out = np.zeros(segmentation.shape)
        for i in range(len(values)):
            out[segmentation == i] = values[i]
        return out

    m = fill_segmentation(shap_values[0][0], segments_slic)

    # code for optionally showing and saving our explanations (turned off)
    # turn on for main.py if want to generate new images - not needed for analysis

    # fig, axes = pl.subplots(nrows=1, ncols=1, figsize=(6,6))
    # max_val = np.max([np.max(np.abs(shap_values[i][:,:-1])) for i in range(len(shap_values))])
    # axes.imshow(img.convert('LA'), alpha=0.15)
    # im = axes.imshow(m, cmap=cm, vmin=-max_val, vmax=max_val)
    # axes.axis('off')
    # cb = fig.colorbar(im, label="SHAP value", aspect=60)
    # cb.outline.set_visible(False)
    # pl.savefig('shaps-{}'.format(index))
    # pl.show()  # heatmap

    # generate pixel saliency values for statistical analysis
    pixel_values = m
    mapped_values = []
    for i in range(pixel_values.shape[0]):
        row = i
        for j in range(pixel_values.shape[1]):
            col = j
            mapped_values.append((row, col, pixel_values[i][j]))
    dict_values = dict(((row, col), val) for row, col, val in mapped_values)

    return dict_values

