# importing libraries
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# loading dataset
my_fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = my_fashion_mnist.load_data()
calss_names = ['Tshirt top', 'Trouser', 'Pullover', 'Dress', 'Cot',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
# to check dataset
print(train_images.shape)
print(len(train_labels))
print(test_images.shape)

# to display data example
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

# preprocessing
train_images = train_images/255.0
test_images = test_images/255.0

# create model
model = keras.Sequential([keras.layers.Flatten(input_shape=(28,28)),
                          keras.layers.Dense(128, activation='relu'),
                          keras.layers.Dense(10, activation='softmax')
                          ])

# compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# train the model
model.fit(train_images, train_labels, epochs=5)

# get the report 
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print("The accuracy is: ", test_accuracy)

# chack the model
predicted_data = model.predict(test_images)
print(predicted_data[0]) # here you can see predicted image as array form

labelled_predicted_data = np.nanargmax(predicted_data)
print("Here is predicted image with label encoded from", labelled_predicted_data)

actual_data = test_labels[0]
print("Here is actual data image with label encoded form", actual_data)