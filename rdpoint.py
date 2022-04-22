import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score



data = pd.read_csv("data_diem3.csv")

data = data.dropna(axis=0)

#X_sequence = data.loc[:, "4/9/2022 10:54:53.899 AM":"4/9/2022 10:55:53.582 AM"].values
#X_sequence = data.loc[:, "3/10/2022 4:24:12.615 PM":"3/10/2022 4:25:12.191 PM"].values
X_sequence = data.loc[:, "4/9/2022 10:54:53.899 AM":"4/9/2022 10:55:53.582 AM"].values
y= data['class'].values
kfold = StratifiedKFold(n_splits=5)

#X_train,X_test,y_train,y_test=train_test_split(X_sequence, y,stratify = y, test_size=0.2)
cvscores=[]
for train,test in kfold.split(X_sequence, y):
    classifier = RandomForestClassifier(n_estimators = 200, criterion = 'entropy', random_state = 0)
    classifier.fit(X_sequence[train],y[train])
    y_pre = classifier.predict(X_sequence[test])
    acc = accuracy_score(y[test], y_pre)
    cvscores.append(acc * 100)
print("mean acc: ",np.mean(cvscores))
# from sklearn.metrics import confusion_matrix
# cf = confusion_matrix(y_test, y_pre)
# print(cf)