import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

classifier = RandomForestClassifier(n_estimators = 200, criterion = 'entropy', random_state = 0)

data = pd.read_csv("data.csv")

data = data.dropna(axis=0)

X_sequence = data.loc[:, "3/10/2022 4:24:12.615 PM":"3/10/2022 4:25:12.191 PM"].values
y= data['class'].values

X_train,X_test,y_train,y_test=train_test_split(X_sequence, y,stratify = y, test_size=0.2)

classifier.fit(X_train,y_train)

y_pre = classifier.predict(X_test)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_pre)
print("acc: ",acc)
from sklearn.metrics import confusion_matrix
cf = confusion_matrix(y_test, y_pre)
print(cf)
