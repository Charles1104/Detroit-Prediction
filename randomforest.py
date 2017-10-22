import pandas as pd
import numpy as np
import cleaning as cl
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

df_train = pd.read_csv('train.csv', encoding = "ISO-8859-1")
df_test = pd.read_csv('test.csv', encoding = "ISO-8859-1")

address =  pd.read_csv('addresses.csv')
latlons = pd.read_csv('latlons.csv')

X_train, X_test, y_train = cl.clean(df_train, df_test, address, latlons)

X_train.info()
#still need to clean violation code

X_train.shape
y_train.shape

grid_values = {'learning_rate': [0.01, 0.1, 1], 'max_depth': [3, 4, 5]}
clf = GradientBoostingClassifier(random_state = 0)
grid_clf_auc = GridSearchCV(clf, param_grid = grid_values, scoring = 'roc_auc')
grid_clf_auc.fit(X_train, y_train)

probs = grid_clf_auc.predict_proba(test_data)[:, 1]
result = pd.Series(probs, index=test_data.index)
