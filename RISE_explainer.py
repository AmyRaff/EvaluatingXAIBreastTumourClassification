import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model
from tqdm import tqdm
from skimage.transform import resize


def riseExplainer(index, Xset, yset):
    # best model
    model = load_model('CNN_75.h5')

    # default vars
    batch_size = 100
    N = 2000
    s = 8
    p1 = 0.5

    ps = model.predict(Xset)
    if ps[index] > 0.5 and yset[index] == 1:
        print('Correctly Classified as Malignant!')
        print(ps[index])
    elif ps[index] < 0.5 and yset[index] == 0:
        print('Correctly Classified as Benign!!')
        print(ps[index])
    elif ps[index] > 0.5 and yset[index] == 0:
        print('Incorrect Classification! Model output is Malignant, but image is Benign.')
        print(ps[index])
    else:
        print('Incorrect Classification! Model output is Benign, but image is Malignant.')
        print(ps[index])
    test_image = Xset[index]

    # generates random masks for image
    def generate_masks(N, s, p1):
        cell_size = np.ceil(np.array((227, 227)) / s)
        up_size = (s + 1) * cell_size
        grid = np.random.rand(N, s, s) < p1
        grid = grid.astype('float32')
        masks = np.empty((N, 227, 227))

        for i in tqdm(range(N), desc='Generating masks'):
            # Random shifts
            x = np.random.randint(0, cell_size[0])
            y = np.random.randint(0, cell_size[1])
            # Linear upsampling and cropping
            masks[i, :, :] = resize(grid[i], up_size, order=1, mode='reflect',
                                    anti_aliasing=False)[x:x + 227, y:y + 227]
        masks = masks.reshape(-1, 227, 227, 1)
        return masks

    # elementwise multiplies masks with original image to get saliencies
    def explain(model, inp, masks):
        preds = []
        # Make sure multiplication is being done for correct axes
        masked = inp * masks
        for i in tqdm(range(0, N, batch_size), desc='Explaining'):
            preds.append(model.predict(masked[i:min(i+batch_size, N)]))
        preds = np.concatenate(preds)
        sal = preds.T.dot(masks.reshape(N, -1)).reshape(-1, 227, 227)
        sal = sal / N / p1
        return sal

    masks = generate_masks(N, s, p1)
    sal = explain(model, test_image, masks)

    # code for showing and saving explanations (turned off)
    # NOTE: causes problems when run for multiple images - colorbar seems to stack.
    # Turn this whole section on for running main.py if want new images, not just the save line
    # Not needed for analysis

    # #plt.axis('off')
    # plt.figure(figsize=(4, 4))
    # plt.imshow(test_image)
    # plt.imshow(sal[0], cmap='jet', alpha=0.6)
    # plt.colorbar()
    # plt.savefig('rise-explanation-{}'.format(index))
    # plt.show()

    # Generate pixel saliency values for statistical analysis
    pixel_values = sal[0]
    mapped_values = []
    for i in range(pixel_values.shape[0]):
        row = i
        for j in range(pixel_values.shape[1]):
            col = j
            mapped_values.append((row, col, pixel_values[i][j]))
    dict_values = dict(((row, col), val) for row, col, val in mapped_values)

    return dict_values
