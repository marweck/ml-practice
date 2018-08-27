import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

print('tensorflow v' + tf.__version__)

# load image data
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images,
                               test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)

# rescale the input data
train_images = train_images / 255.0
test_images = test_images / 255.0

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.cmap_d['binary'])
    plt.xlabel(class_names[train_labels[i]])

# prepares the model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(), loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# train the model
model.fit(train_images, train_labels, epochs=5)

# evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

# test the model
predictions = model.predict(test_images)
print('Prediction 0: ', predictions[0])
print('Test 0: ', np.argmax(predictions[0]), ' <=> ', test_labels[0])

# test the model with a single image
img = test_images[0]
img = (np.expand_dims(img, 0))
one_pred = model.predict(img)
print('Test Img 0: ',  np.argmax(one_pred[0]), ' <=> ', test_labels[0])
