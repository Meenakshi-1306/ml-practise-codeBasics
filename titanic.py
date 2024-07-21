import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
data=pd.read_excel(r"C:\Users\MANICKA MEENAKSHI.S\Downloads\titanic.xlsx")
df=pd.DataFrame(data)
#changing the categorical
Sex_n=LabelEncoder()
df['Sex_n']=Sex_n.fit_transform(df['Sex'])
df=df.drop(['PassengerId','Name','Sex','SibSp','Parch','Ticket','Cabin','Embarked'],axis=1)
Y=df.Survived
X=df.drop(['Survived'],axis=1)
Tree=tree.DecisionTreeClassifier()
Tree.fit(X,Y)
scr=Tree.score(X,Y)
print(scr)


