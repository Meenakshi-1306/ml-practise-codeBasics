import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
data=pd.read_excel(r"C:\Users\MANICKA MEENAKSHI.S\Downloads\ml practise\salaries.xlsx")
df=pd.DataFrame(data)
#changing the categorical
com_n=LabelEncoder()
df['com_n']=com_n.fit_transform(df['company'])

job_n=LabelEncoder()
df['job_n']=com_n.fit_transform(df['job'])

degree_n=LabelEncoder()
df['degree_n']=com_n.fit_transform(df['degree'])
df = df.drop(['company', 'job', 'degree'], axis=1)
print(df)
hunK=df['salary_more_then_100k']
Y=hunK
print("Y.....\n")
print(Y)
X=df.drop(['salary_more_then_100k'],axis=1)
Tree=tree.DecisionTreeClassifier()
Tree.fit(X,Y)
scr=Tree.score(X,Y)
print(scr)

