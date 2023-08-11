import numpy as np # linear algebra
import pandas as pd # data processing
import seaborn as sns # data visualization
import matplotlib.pyplot as plt # data visualization
# import warnings
import warnings
# filter warnings
warnings.filterwarnings('ignore')
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from imutils.contours import sort_contours
import imutils # !pip install imutils
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.optimizers import SGD
from keras.optimizers import RMSprop,Adam
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import BatchNormalization
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.regularizers import l1,l2
import os
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical

x = []
y = []
for i in range(0,16):
    for j in range(1,21):
        a = os.path.isfile('allData/dataset/'+str(i)+'/'+str(j)+'.png')
        if a :
            image = cv2.imread('allData/dataset/'+str(i)+'/'+str(j)+'.png', 0)
            img_resized = cv2.resize(255-image, (28,28), interpolation=cv2.INTER_AREA)
            unrolled = img_resized.ravel()
            x.append(unrolled)
            y.append(i)

# Converting x and y variables to numpy arrays
x = np.array(x)
y = np.array(y)

# Shapes of x and y arrays
print(x.shape)
print(y.shape)

# train = %70 and test = %30 of overall dataset
x_train, x_val_test, y_train, y_val_test = train_test_split(x, y, test_size=0.3, random_state=123)

x_val, x_test, y_val, y_test = train_test_split(x_val_test, y_val_test, test_size=0.5, random_state=123)

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_val = x_val.reshape(x_val.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1) 

input_shape = (28, 28, 1)

# We convert the values held by our variables to float
x_train = x_train.astype('float32')
x_val = x_val.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_val /= 255
x_test /= 255

print("x_train shape: ",x_train.shape)
print("y_train shape: ",y_train.shape)

# one hot encoding
y_train = to_categorical(y_train, num_classes = 16)

# splitting for fitting
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.1, random_state=123)

# Shapes of x_train, x_test, y_train and y_test
print("x_train shape",x_train.shape)
print("x_test shape",x_val.shape)
print("y_train shape",y_train.shape)
print("y_test shape",y_val.shape)

# See some image
plt.imshow(x_train[2][:,:,0],cmap='gray')
plt.show()

plt.imshow(x_train[0][:,:,0],cmap='gray')
plt.show()

data_generator = ImageDataGenerator(
    rescale = 1./255, 
    shear_range = 0.2, 
    zoom_range = 0.2,
    validation_split = 0.25,
    featurewise_center=True,  # set input mean to 0 over the dataset
    samplewise_center=True,  # set each sample mean to 0
    featurewise_std_normalization=True,  # divide inputs by std of the dataset
    samplewise_std_normalization=True,  # divide each input by its std
    zca_whitening=True,  # dimesion reduction
    rotation_range=0.5,  # randomly rotate images in the range 5 degrees
    width_shift_range=0.5,  # randomly shift images horizontally 5%
    height_shift_range=0.5,  # randomly shift images vertically 5%
    horizontal_flip=True,  # randomly flip images
    vertical_flip=True     # randomly flip images
)

# The location of the data to be used
path = 'allData/dataset'

# Train data
train = data_generator.flow_from_directory(
    path, 
    class_mode = 'categorical',
    target_size = (40, 40),
    subset='training',
    color_mode = 'grayscale',
    batch_size = 10,
    shuffle = True,
    seed = 123
)

# Validation data
valid = data_generator.flow_from_directory(
    path, 
    class_mode = 'categorical',
    target_size = (40, 40), 
    subset='validation',
    color_mode = 'grayscale',
    batch_size = 10,
    shuffle = True,
    seed = 123
)

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu', input_shape = (40,40,1)))

# kernel_regularizer=l1(0.01),activity_regularizer=l2(0.01)

model.add(MaxPool2D(pool_size=(2,2)))
# model.add(Dropout(0.25))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

# model.add(Dropout(0.25))

# fully connected

model.add(Flatten())
model.add(Dense(128, activation = "relu"))
# model.add(Dropout(0.5))
model.add(Dense(16, activation = "softmax"))

#model.add(BatchNormalization())

# Define the optimizer
optimizer = Adam(lr = 5e-4)

# Compile the model
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])

epochs = 100
batch_size = 10

model.summary()

# fit the model
hist = model.fit_generator(train,validation_data=valid,epochs=100 ,verbose=1,validation_steps = 10)

# Calculate the accuracy
val_loss, val_accuracy = model.evaluate(valid)
print(val_loss,val_accuracy)

print(hist.history.keys())
plt.plot(hist.history["loss"], label = "Train Loss")
plt.legend()
plt.show()
plt.figure()
plt.plot(hist.history["accuracy"], label = "Train acc")
plt.legend()
plt.show()

# train set classes
train.class_indices

label = (train.class_indices)
label

def predict_image(image):

    plt.imshow(image, cmap = 'gray')
    image = cv2.resize(image,(40, 40))
    normalized = cv2.normalize(image, None, alpha = 0, beta = 1, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    normalized = normalized.reshape((normalized.shape[0], normalized.shape[1], 1))
    y = np.asarray([normalized])
    prediction = np.argmax(model.predict(y))
    
    return prediction

image = cv2.imread('2.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

edged = cv2.Canny(blurred, 30, 150)
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sort_contours(cnts, method="left-to-right")[0]
chars=[]

for c in cnts:

    (x, y, w, h) = cv2.boundingRect(c)

    if w*h>1200:
       
        roi = gray[y:y + h, x:x + w]
        chars.append(predict_image(roi))
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

plt.figure(figsize=(10,10))
plt.imshow(image)

# characters in the picture
chars

labels=[i for i in train.class_indices]
print(labels)

labels[2] = "-"
labels[3] = "+"
labels[4] = "*"
labels[5] = "/"
labels[6] = "["
labels[7] = "]"

equation = []
result = ""
for i in chars:
    equation.append(labels[i])
    result = ''.join(equation)

print(result)