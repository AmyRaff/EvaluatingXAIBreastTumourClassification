from tensorflow.keras.layers import Conv2D, Input, ZeroPadding2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model
import helper

# get data splits
X_train, y_train = helper.X_train, helper.y_train
X_val, y_val = helper.X_val, helper.y_val
X_test, y_test = helper.X_test, helper.y_test

print("number of training examples = " + str(X_train.shape[0]))
print("number of development examples = " + str(X_val.shape[0]))
print("number of testing examples = " + str(X_test.shape[0]))


# build our CNN - taken directly from https://github.com/MohamedAliHabib/Brain-Tumor-Detection, only renamed
def build_model(input_shape):
    X_input = Input(input_shape)
    # Zero-Padding: pads the border of X_input with zeroes
    X = ZeroPadding2D((2, 2))(X_input)
    # CONV -> BN -> RELU Block applied to X
    X = Conv2D(32, (7, 7), strides=(1, 1), name='conv0')(X)
    X = BatchNormalization(axis=3, name='bn0')(X)
    X = Activation('relu')(X)
    # MAXPOOL
    X = MaxPooling2D((4, 4), name='max_pool0')(X)
    # MAXPOOL
    X = MaxPooling2D((4, 4), name='max_pool1')(X)
    # FLATTEN X
    X = Flatten()(X)
    # FULLYCONNECTED
    X = Dense(1, activation='sigmoid', name='fc')(X)
    # Create model. This creates your Keras model instance, use this instance to train/test the model.
    model = Model(inputs=X_input, outputs=X, name='BreastDetectionModel')
    return model


# generate a model save for use with explanation code
IMG_SHAPE = (227, 227, 3)
model = build_model(IMG_SHAPE)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# train the model with various numbers of epochs as detailed in report
history = model.fit(X_train, y_train, epochs=100, verbose=1)
model.save('CNN_100.h5')