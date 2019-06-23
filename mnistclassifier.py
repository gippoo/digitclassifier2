import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPool2D
from keras.preprocessing.image import ImageDataGenerator

# Load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values to be between 0 and 1
x_train = x_train/255
x_test = x_test/255

# Examine random images to see if data loaded properly
plt.figure(figsize=(9,9))
for i in range(25):
    rn = np.random.randint(0,len(x_train))
    plt.subplot(5, 5, i+1)
    plt.imshow(x_train[rn])
    plt.title('Class {}'.format(y_train[rn]))
    plt.axis('off')

# Prepare data for training
# Turn class labels into 1 hot vectors
nb_classes = 10
y_train = to_categorical(y_train, nb_classes)
y_test = to_categorical(y_test, nb_classes)

x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)

# Build the model
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5,5), padding='Same', activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(filters=32, kernel_size=(5,5), padding='Same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters=64, kernel_size=(3,3), padding='Same', activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3,3), padding='Same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

model.summary()

# Apply data augmentation
datagen = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=False,
        vertical_flip=False)

datagen.fit(x_train)

# Train the model
model.fit_generator(datagen.flow(x_train,y_train, batch_size=64),
                    epochs=2,
                    validation_data=(x_test, y_test))


# Check how the model performs on random images in the test set
plt.figure(figsize=(9,9))
for i in range(25):
    rn = np.random.randint(0,len(x_test))
    plt.subplot(5, 5, i+1)
    plt.imshow(x_test[rn].reshape(28,28))
    plt.title('Pred: {}'.format(np.argmax(model.predict(x_test[rn].reshape(-1,28,28,1)))))
    plt.axis('off')