import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')

seed = 7
np.random.seed(seed)

ytrain = np.loadtxt(open("digit-recognizer/train.csv",'rb'),delimiter=',',skiprows=1,usecols=0,max_rows=40000)
xtrain = np.loadtxt(open('digit-recognizer/train.csv','rb'),delimiter=',',skiprows=1,usecols=range(784),max_rows=40000)
ytest = np.loadtxt(open('digit-recognizer/train.csv','rb'),delimiter=',',skiprows=40001,usecols=0)
xtest = np.loadtxt(open('digit-recognizer/train.csv','rb'),delimiter=',',skiprows=40001,usecols=range(784))

X_train = xtrain.reshape(xtrain.shape[0], 1, 28, 28).astype('float32')
X_test = xtest.reshape(xtest.shape[0], 1, 28, 28).astype('float32')

X_train = X_train / 255
X_test = X_test / 255
# one hot encode outputs
y_train = np_utils.to_categorical(ytrain)
y_test = np_utils.to_categorical(ytest)
num_classes = y_test.shape[1]

def baseline_model():
    # create model
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# build the model
model = baseline_model()
# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=40, batch_size=50, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy Score: %.2f%%" % (scores[1]*100))