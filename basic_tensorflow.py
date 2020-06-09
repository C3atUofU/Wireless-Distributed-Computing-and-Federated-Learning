import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

mnist = tf.keras.datasets.mnist # mnist is a dataset of 28x28 images of hand-written digits 0-9
(x_train, y_train), (x_test, y_test) = mnist.load_data() # unpacks images to x_train/x_test and labels to y_train/y_test

# Normalize. Make network easy to learn
x_train = tf.keras.utils.normalize(x_train, axis = 1) # scales data between 0 and 1
# (x: Numpy array to normalize, axis: axis along which to normalize)
x_test = tf.keras.utils.normalize(x_test, axis = 1) # scales data between 0 and 1

# Build the model
model = tf.keras.models.Sequential() # most common one. A basic feed-forward model

model.add(tf.keras.layers.Flatten()) # Input layer. Takes our 28x28 and makes it 1x784
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) # First hidden layer with 128 neurons, relu activation
# tf.nn.relu is default activation function
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) # Second hidden layer with 128 neurons, relu activation
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax)) # Output layer. 10 units for 10 classes. Softmax for probability distribution
#  tf.nn.softmax is for a probability distribution

# Define some parameters for the training of the model
model.compile(optimizer='adam', # Good default optimizer to start with
              loss='sparse_categorical_crossentropy', # how will we calculate our "error." Neural network aims to minimize loss.
              metrics=['accuracy'])  # what to track

# Train the model
model.fit(x_train, y_train, epochs=3)
val_loss, val_acc = model.evaluate(x_test, y_test) # evaluate the out of sample data with model
print(val_loss, val_acc) # model's loss (error) and accuracy

predictions = model.predict([x_test]) # predict always takes a list
print(np.argmax(predictions[1]))
plt.imshow(x_test[1], cmap=plt.cm.binary)
plt.show()

#plt.imshow(x_train[0], cmap=plt.cm.binary)
#plt.show()
#print(x_train[0])