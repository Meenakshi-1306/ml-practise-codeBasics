import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.compose import ColumnTransformer
from openpyxl import load_workbook  # Import openpyxl for reading Excel files

# Assuming the file path is correct, load the data from Excel
data = pd.read_excel(r"C:\Users\MANICKA MEENAKSHI.S\Downloads\ml practise\car_OneHotEncoding.xlsx")
df = pd.DataFrame(data)

# Display the scatter plot (uncomment if desired)
plt.scatter(df.Age, df.Price)
# plt.show()

X = df[['CarModel', 'Mileage', 'Age']]
Y=df['Price']
# One-hot encode the 'CarModel' feature (assuming it's categorical)
from sklearn.preprocessing import OneHotEncoder
column_transformer = ColumnTransformer(
    transformers=[
        ('encoder', OneHotEncoder(), ['CarModel'])
    ],
    remainder='passthrough'
)
X = column_transformer.fit_transform(X)
X=X[:, 1:]
reg=linear_model.LinearRegression()
ref=reg.fit(X,Y)
pred=reg.predict([[1,0,86000,7]])
print(pred)
scr=reg.score(X,Y)
print(scr)
import joblib
joblib.dump(reg, 'Car price model save.pkl')

joblib.dump(column_transformer, 'column_transformer.pkl')

mj=joblib.load('Car price model save.pkl')
loaded_column_transformer = joblib.load('column_transformer.pkl')
#print(X)
sans=mj.predict([[1,0,86000,7]])
print(sans)

