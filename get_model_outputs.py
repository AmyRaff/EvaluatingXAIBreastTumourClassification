from collections import Counter

from tensorflow.keras.models import load_model
import helper
from sklearn import metrics
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# This file is used to evaluate model performances

model = load_model('CNN_75.h5')
IMG_WIDTH, IMG_HEIGHT = (227, 227)

# toggle data as needed
#val_X = helper.X_val
#val_y = helper.y_val
test_X = helper.X_test
test_y = helper.y_test

# calculate accuracy
loss, acc = model.evaluate(x=test_X, y=test_y)
preds = model.predict(test_X)
predicted_labels = []

for i in range(len(preds)):
    outs = (test_y[i], preds[i])
    if preds[i] > 0.5:
        predicted_labels.append(1)
    else:
        predicted_labels.append(0)
    print(outs)

# calculate F1 Scores
f1_score = helper.compute_f1_score(test_y, predicted_labels)
print("F1 Score: {}".format(f1_score))
print("num_mal = {}".format(Counter(predicted_labels)[1]))
print("num_ben = {}".format(Counter(predicted_labels)[0]))


# generate, plot and save a confusion matrix
cm = metrics.confusion_matrix(test_y, predicted_labels)
ax = plt.subplot()
sns.heatmap(cm, annot=True, cmap="YlGnBu")

ax.set_xlabel("Predicted Labels")
ax.set_ylabel("True Labels")
ax.set_title("Confusion Matrix")
ax.xaxis.set_ticklabels(['Benign', 'Malignant'])
ax.yaxis.set_ticklabels(['Benign', 'Malignant'])
plt.savefig("75-test-end-model-confmap.png")
plt.show()
