
# Importing Packages
import numpy as np
import seaborn as sns
from keras import regularizers
from sklearn import metrics
from tensorflow.keras.optimizers import Adam
import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Loading Dataset
from keras.datasets import fashion_mnist


(x_train, y_train),(x_test, y_test) = fashion_mnist.load_data()

print("train set: ", x_train.shape, ",", y_train.shape, "Test set: ", x_test.shape, ",", y_test.shape)


plt.imshow(x_train[0])
plt.title('Class: {}'.format(y_train[0]))
plt.figure()

# Normalizing the Data
x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis = 1)

plt.imshow(x_train[0])
plt.title('Class: {}'.format(y_train[0]))
plt.figure()

print(x_train.shape)

# Defining the Model
model = Sequential()
model.add(Flatten(input_shape=((28,28))))
model.add(Dense(200, kernel_regularizer=regularizers.l2(0.0001), activation="relu"))
model.add(Dense(10, kernel_regularizer=regularizers.l2(0.0001), activation="softmax"))

# Compiling the Model
model.compile(optimizer=Adam(learning_rate=0.001), loss="sparse_categorical_crossentropy", metrics = ["accuracy"])

# Fitting the Model
history = model.fit(x_train, y_train, epochs = 20, batch_size = 1000, validation_split = 0.15,)

history_dict = history.history

# Evaluating on the Test Data
val_loss, val_acc = model.evaluate(x_test, y_test)

print("Test Loss: ", val_loss, " Test Accuracy: ", val_acc)

model.summary()
print(history.history.keys())
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()

accuracy_values = history_dict['accuracy']
val_accuracy_values=history_dict['val_accuracy']
plt.plot(val_accuracy_values,'-r',label='val_accuracy')
plt.plot(accuracy_values,'-b',label='accuracy')
plt.legend()
plt.show()

