import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
ir= load_iris()
print(dir(ir))
X_train, X_test, Y_train, Y_test = train_test_split(ir.data, ir.target, test_size=0.3)
fore=RandomForestClassifier(n_estimators=405060)
fore.fit(X_train,Y_train)
scr=fore.score(X_test,Y_test)
print(scr)