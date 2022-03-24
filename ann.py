
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score

print(tf.__version__)
data = pd.read_csv("/home/fit/Pictures/ANHHAI-VIENNGHIENCUU/DORUNGDONGCO/Ket qua thu nghiem/data.csv")
data = data.dropna(axis=0)
X_sequence = data.loc[:, "3/10/2022 4:24:12.615 PM":"3/10/2022 4:25:12.191 PM"].values
y= data['class'].values


X_train,X_test,y_train,y_test=train_test_split(X_sequence, y,stratify = y, test_size=0.2)

print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#----------------------- Building the model -----------------------#

ann = tf.keras.models.Sequential()

ann.add(tf.keras.layers.Dense(units = 238, activation = 'relu'))

ann.add(tf.keras.layers.Dense(units = 114, activation = 'relu'))
ann.add(tf.keras.layers.Dense(units = 72, activation = 'relu'))
ann.add(tf.keras.layers.Dense(units = 6, activation = 'relu'))


ann.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))

#----------------------- Training the model -----------------------#
# Compiling the ANN
# Type of Optimizer = Adam Optimizer, Loss Function =  crossentropy for binary dependent variable, and Optimization is done w.r.t. accuracy
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Training the ANN model on training set  (fit method always the same)
# batch_size = 32, the default value, number of epochs  = 100
ann.fit(X_train, y_train, batch_size =64, epochs = 1000)

#----------------------- Evaluating the Model ---------------------#

y_pred_prob = ann.predict(X_test)

#probabilities to binary
y_pred = (y_pred_prob > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)), 1))

#Confusion Matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix", confusion_matrix)
print("Accuracy Score", accuracy_score(y_test, y_pred))