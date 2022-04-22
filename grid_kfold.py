import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3, 4, 5, 6],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [50,100,200,300,400,500]
}
def evaluate(model, test_features, test_labels):
    #predictions = model.predict(test_features)
    y_pre = model.predict(test_features)

    from sklearn.metrics import accuracy_score
    acc = accuracy_score(test_labels, y_pre)
    print("acc: ",acc)
    from sklearn.metrics import confusion_matrix
    cf = confusion_matrix(test_labels, y_pre)
    print(cf)
    return acc

data = pd.read_csv("data_all.csv")

data = data.dropna(axis=0)

X_sequence = data.loc[:, "3/10/2022 4:24:12.615 PM":"3/10/2022 4:25:12.191 PM"].values
y= data['class'].values
kfold = StratifiedKFold(n_splits=5)

#X_train,X_test,y_train,y_test=train_test_split(X_sequence, y,stratify = y, test_size=0.2)
cvscores=[]
for train,test in kfold.split(X_sequence, y):
    classifier = RandomForestClassifier(n_estimators = 200, criterion = 'entropy', random_state = 0)
    grid_search = GridSearchCV(estimator = classifier, param_grid = param_grid, 
                          cv = 5, n_jobs = -1, verbose = 2, return_train_score=True)
    grid_search.fit(X_sequence[train],y[train])
    #y_pre = classifier.predict(X_sequence[test])
    best_grid = grid_search.best_estimator_
    grid_accuracy = evaluate(best_grid, X_sequence[test], y[test])
    #acc = accuracy_score(y[test], y_pre)
    cvscores.append(grid_accuracy * 100)
print("mean acc: ",np.mean(cvscores))
# from sklearn.metrics import confusion_matrix
# cf = confusion_matrix(y_test, y_pre)
# print(cf)