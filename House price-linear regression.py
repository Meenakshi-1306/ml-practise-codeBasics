import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

# Load the Excel file
data = pd.read_excel(r"C:\Users\MANICKA MEENAKSHI.S\Downloads\ml practise\linear.xlsx")

# Create DataFrame
df = pd.DataFrame(data)

# Plotting the data
plt.title("HOUSE PRICING USING LINEAR REGRESSION")
plt.xlabel("Area")
plt.ylabel("Price")
plt.scatter(df['area'], df['price'], color='red', marker="+")

# Performing linear regression
reg = linear_model.LinearRegression()
reg.fit(df[['area']], df['price'])

# Making predictions
amt = reg.predict(df[['area']])

# Creating a new column for predictions
df['prediction'] = amt

# Displaying the predictions and the DataFrame
print(df)

# Plotting the regression line
plt.plot(df['area'], df['prediction'], color='blue')

# Show plot
plt.show()
