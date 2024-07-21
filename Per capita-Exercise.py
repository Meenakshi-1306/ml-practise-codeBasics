import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load the Excel file
data = pd.read_excel(r"C:\Users\MANICKA MEENAKSHI.S\OneDrive\Documents\Copy of canada_per_capita_income(1).xlsx")
df = pd.DataFrame(data)

# Display the DataFrame
#print(data)

# Plotting the data
plt.scatter(df.year, df.perCapita, marker='+', color='red')
plt.xlabel('Year')
plt.ylabel('Per Capita Income')
plt.title('Year vs Per Capita Income')
plt.show()

# Performing linear regression
reg = LinearRegression()
reg.fit(df[['year']], df.perCapita)

# Making predictions
predicted_income = reg.predict([[2050]])
print(f"Predicted per capita income for the year 2050: {predicted_income[0]}")

# Print the R-squared value
r_squared = reg.score(df[['year']], df.perCapita)
print(f"R-squared value: {r_squared}")
