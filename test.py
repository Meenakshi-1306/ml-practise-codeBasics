import pandas as pd
import joblib

# Load the saved model and column transformer
mj = joblib.load('Car price model save.pkl')
loaded_column_transformer = joblib.load('column_transformer.pkl')
sans=mj.predict([[1,0,86000,7]])
print(sans)

