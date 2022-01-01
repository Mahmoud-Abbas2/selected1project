import tensorflow as tf

import seaborn as sns

import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
import keras

import numpy as np

from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras import regularizers
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
import pandas as pd

import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import matplotlib
from matplotlib import pyplot as plt

# Import input (x) and output (y) data, and asign these to df1 and df2
X = pd.read_csv('F:\X_data.csv')
Y = pd.read_csv('F:\Y_data.csv')

X = preprocessing.scale(X)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=2)

print(X_train.shape)

def buildModel():
    model = Sequential()
    model.add(Dense(units = 16, kernel_regularizer=regularizers.l2(0.001), activation='sigmoid', input_dim = 30))
    model.add(Dense(units = 16, kernel_regularizer=regularizers.l2(0.001), activation='sigmoid'))
    model.add(Dense(units = 1, kernel_regularizer=regularizers.l2(0.001), activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=0.01), loss= 'binary_crossentropy', metrics= ['accuracy'])
    return model

model = buildModel()

earlystopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=1, mode='auto')

history = model.fit(X_train, y_train, epochs = 100, batch_size = 100, validation_split = 0.15, verbose = 0, callbacks = [earlystopper])

history_dict=history.history

val_loss, val_acc = model.evaluate(X_test, y_test)


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

# get predictions
y_pred = model.predict(X_test, verbose=2)
print("Aho",y_pred.shape)
print("Aho",y_train.shape)
# compute confusion matrix with `tf` 
confusion = tf.math.confusion_matrix(
              labels = np.argmax(y_test, axis=1),       
              predictions = np.argmax(y_pred, axis=1),   
              num_classes=2)

print(confusion)               

cm = pd.DataFrame(confusion.numpy(), # use .numpy(), because now confusion is tensor
               range(2),range(2))

plt.figure(figsize = (2,2)) 
sns.heatmap(cm, annot=True, annot_kws={"size": 12}) # font size
plt.show()

from sklearn.metrics import roc_curve
from sklearn.metrics import auc
y_test_pred = model.predict(X_test)
fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test,y_test_pred)
auc_keras = auc(fpr_keras, tpr_keras)
print('Testing data AUC: ', auc_keras)
# ROC curve of testing data

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
# plt.plot(fpr_rf, tpr_rf, label='RF (area = {:.3f})'.format(auc_rf))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()


loss, acc = model.evaluate(X_test, y_test)
print("Test loss: ", loss)
print("Test accuracy: ", acc)

classifier = KerasClassifier(build_fn= buildModel, batch_size = 100, verbose = 0, epochs = 100)
accuracies = cross_val_score(estimator= classifier, X = X, y = Y, cv=4)

print("Accuracies:\n", accuracies)
print("\n")
print("Mean Accuracy: ", accuracies.mean())
    












