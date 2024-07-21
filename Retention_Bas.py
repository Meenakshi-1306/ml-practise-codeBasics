import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.compose import ColumnTransformer
from openpyxl import load_workbook 
data=pd.read_excel(r"C:\Users\MANICKA MEENAKSHI.S\Downloads\ml practise\HR_comma_sep.xlsx")
df=pd.DataFrame(data)
#print(df.head(3))
pd.crosstab(df.salary,df.left).plot(kind='bar')
#plt.show()

pd.crosstab(df.Department,df.left).plot(kind='bar')
#plt.show()
X=df[['satisfaction_level','number_project','average_montly_hours','salary','promotion_last_5years','last_evaluation']]
from sklearn.preprocessing import OneHotEncoder
column_transformer = ColumnTransformer(
    transformers=[
        ('encoder', OneHotEncoder(), ['salary'])
    ],
    remainder='passthrough'
)
X = column_transformer.fit_transform(X)
X=X[:, 1:]
print(X.shape)
print(X)
Y=df[['left']]
#print()

log=linear_model.LogisticRegression()
log.fit(X,Y)
scr=log.score(X,Y)
print(scr)