import tensorflow.keras as keras
from tensorflow.keras.layers import LSTM, Dropout, Dense
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

data = pd.read_csv("/home/fit/Pictures/ANHHAI-VIENNGHIENCUU/DORUNG/Ket qua thu nghiem/data.csv")
data=data.dropna()
print(data.head())
sequence_length = 1

# scaler = MinMaxScaler()
# data.loc[:, "3/10/2022 4:24:12.615 PM":"3/10/2022 4:25:12.191 PM"] = scaler.fit_transform(data.loc[:, "3/10/2022 4:24:12.615 PM":"3/10/2022 4:25:12.191 PM"])

def generate_data(X, y, sequence_length = 1, step = 1):
    X_local = []
    y_local = []
    for start in range(0, len(data) - sequence_length, step):
        end = start + sequence_length
        X_local.append(X[start:end])
        y_local.append(int(y[end-1]))
    X_local = np.array(X_local)
    #X_local = X_local.reshape(-1,238,1)
    return X_local, np.array(y_local)

X_sequence, y = generate_data(data.loc[:, "3/10/2022 4:24:12.615 PM":"3/10/2022 4:25:12.191 PM"].values, data['class'].values)

#X_sequence = scaler.fit_transform(X_sequence)
#show data
print(X_sequence)
print(y)

model = keras.Sequential()
#model.add(LSTM(100, input_shape = (238, 1)))
model.add(LSTM(100, input_shape = (1,238)))
model.add(Dropout(0.5))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss="binary_crossentropy"
              , metrics=[keras.metrics.binary_accuracy]
              , optimizer="adam")

model.summary()

training_size = int(len(X_sequence) * 0.8)
X_train,X_test,y_train,y_test=train_test_split(X_sequence, y,stratify = y, test_size=0.2,random_state=0)

print("sum: ",y_test.sum())
print("len: ",len(y_test))

model.fit(X_train, y_train, batch_size=32, epochs=100)
model.evaluate(X_test, y_test)
y_test_prob = model.predict(X_test, verbose=1)
y_test_pred = np.where(y_test_prob > 0.5, 1, 0)
from sklearn.metrics import confusion_matrix
cf = confusion_matrix(y_test, y_test_pred)
print(cf)
