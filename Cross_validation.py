import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier 
from sklearn import linear_model
from sklearn.svm import SVC
from sklearn import tree

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
ir= load_iris()
print(dir(ir))
X_train, X_test, Y_train, Y_test = train_test_split(ir.data, ir.target, test_size=0.3)
a=cross_val_score(RandomForestClassifier(),X_train,Y_train,cv=5)
a1=cross_val_score(linear_model.LinearRegression(),X_train,Y_train,cv=5)
a2=cross_val_score(linear_model.LogisticRegression(),X_train,Y_train,cv=5)
a3=cross_val_score(tree.DecisionTreeClassifier(),X_train,Y_train,cv=5)
s_scores = cross_val_score(SVC(), X_train,Y_train,cv=5)

print(a,a1,a2,a3)
rf=np.average(a)
lr=np.average(a1)
logi=np.average(a2)
trees=np.average(a3)
svmm=np.average(s_scores)
print("Scores....")
print(rf,lr,logi,trees,svmm)
