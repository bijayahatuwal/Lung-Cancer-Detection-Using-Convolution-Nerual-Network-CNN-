
import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np
from keras import backend as k
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Conv2D, MaxPool2D
from keras.models import Sequential
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import joblib

# dimension of the image
img_width, img_height = 180, 180

train_data_dir = r"/content/lung_image_sets_01/train"
validation_data_dir = r'/content/lung_image_sets_01/valid'
test_data_dir = r'/content/lung_image_sets_01/valid'

train_model = True
test_model = False

nb_train_samples = 0
nb_validation_samples = 0
nb_test_samples = 0
epochs = 0
batch_size = 64
class_number = 3

if train_model:
    if os.path.isdir(train_data_dir):
        for r, d, f in os.walk(train_data_dir):
            nb_train_samples = nb_train_samples + len(f)

    if os.path.isdir(validation_data_dir):
        for r, d, f in os.walk(validation_data_dir):
            nb_validation_samples = nb_validation_samples + len(f)

    if os.path.isdir(validation_data_dir):
        for r, d, f in os.walk(validation_data_dir):
            nb_test_samples = nb_validation_samples + len(f)

    train_epochs = math.ceil(nb_train_samples / batch_size)

    if k.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)

train_datagen = ImageDataGenerator(
        rescale = 1./255,
        horizontal_flip = True,
        vertical_flip=False,
        zoom_range=0.2
    )

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

validation_datagen = ImageDataGenerator(rescale=1. / 255)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

#############################################

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Flatten())
# model.add(Dense(64))
model.add(Dense(36))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Dense(class_number))  # number of output i.e file name in our case
model.add(Activation('sigmoid'))

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',  # 'rmsprop',
    metrics=['accuracy']
)

# this is the augmentation configuration we eill use for training
# Fit the model

model.summary()

history = model.fit_generator(
    train_generator,
    steps_per_epoch=math.ceil(nb_train_samples / batch_size),
    epochs = 20,
    validation_data=validation_generator,
    validation_steps= math.ceil(nb_validation_samples / batch_size)
)

model.save('rough_plant_01.h5')

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()
time.sleep(5)

from sklearn.metrics import confusion_matrix, classification_report
# This is the augmentation configuration we will use for testing
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)
#Confution Matrix and Classification Report
test_batch_size = 50
Y_pred = model.predict_generator(test_generator, nb_test_samples // test_batch_size+1)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix', len(y_pred))
print(confusion_matrix(test_generator.classes, y_pred))
print('Classification Report')
target_names = ['adenocarcinoma', 'benign_tissue','squamous_cell_carcinoma']
print(classification_report(test_generator.classes, y_pred, target_names=target_names))

import itertools
#Confution Matrix and Classification Report
# steps_per_epoch = math.ceil(nb_train_samples / batch_size)
Y_pred = model.predict_generator(test_generator, nb_test_samples // test_batch_size+1)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix', len(y_pred))
print(confusion_matrix(test_generator.classes, y_pred))
print('Classification Report')
target_names = ['adenocarcinoma', 'benign_tissue','squamous_cell_carcinoma']
print(classification_report(test_generator.classes, y_pred, target_names=target_names))

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# cm = confusion_matrix(test_labels, predictions)
cm = confusion_matrix(test_generator.classes, y_pred)
cm_plot_labels = ['adenocarcinoma', 'benign_tissue','squamous_cell_carcinoma']
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')
