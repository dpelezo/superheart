import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn import metrics
from sklearn import neighbors
from sklearn.svm import SVC


heart = pd.read_csv('clevelanda.csv')
heart.dropna()
heart['ca'] = pd.to_numeric(heart['ca'])
heart['thal'] = pd.to_numeric(heart['thal'])
X = heart.iloc[:,:13]
y = heart.iloc[:,13:]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y , test_size = 0.2)
log_clf = LogisticRegression(solver="lbfgs", random_state=42)
rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, random_state=42)
svm_clf = SVC(gamma="scale", probability=True, random_state=42)
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(), n_estimators=500,
    max_samples=100, bootstrap=True, random_state=42)

voting_clf = VotingClassifier(
    estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf), ('bag', bag_clf)],
    voting='hard')
voting_clf.fit(X_train, y_train)
voting_clf.score(X_train, y_train)
predicted = voting_clf.predict(X_test)
predicted
#Saving Model
import pickle

pickle.dump(voting_clf, open("model.pkl", "wb" ))

my_scaler = pickle.load(open("model.pkl", "rb" ))

predictions = my_scaler.predict(X_test)