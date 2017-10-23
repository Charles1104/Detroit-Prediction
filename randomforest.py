import pandas as pd
import numpy as np
import cleaning as cl
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

df_train = pd.read_csv('datasets/train.csv', encoding = "ISO-8859-1")
df_test = pd.read_csv('datasets/test.csv', encoding = "ISO-8859-1")

address =  pd.read_csv('datasets/addresses.csv')
latlons = pd.read_csv('datasets/latlons.csv')

X_train, X_test, y_train = cl.clean(df_train, df_test, address, latlons)

X_train.info()

X_train.shape
X_test.shape
y_train.shape

grid_values = {'learning_rate': [0.01, 0.1, 1], 'max_depth': [3,5]}
clf = GradientBoostingClassifier(random_state = 0)
grid_clf_auc = GridSearchCV(clf, param_grid = grid_values, scoring = 'roc_auc')
grid_clf_auc.fit(X_train, y_train)

grid_clf_auc.best_score_

grid_values2 = {'n_estimators': [5, 10, 20], 'max_features': [2,3,4], 'max_depth': [3,5]}
clf2 = RandomForestClassifier(random_state = 0)
grid_clf2_auc = GridSearchCV(clf2, param_grid = grid_values2, scoring = 'roc_auc')
grid_clf2_auc.fit(X_train, y_train)

grid_clf2_auc.best_score_

probs = grid_clf_auc.predict_proba(test_data)[:, 1]
result = pd.Series(probs, index=test_data.index)
