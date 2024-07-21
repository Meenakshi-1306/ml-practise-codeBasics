import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load the digits dataset
ir= load_iris()
print(dir(ir))

print(ir.target_names)
# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(ir.data, ir.target, test_size=0.2)

# Print the training data
print(X_train)
log=linear_model.LogisticRegression(max_iter=10000)
log.fit(X_train,Y_train)
src=log.score(X_test,Y_test)
ans=log.predict([ir.data[50]])
print("Score:",src)
print("Predicted:",ans)
plt.scatter(ir.data[:, 0], ir.data[:, 1], c=ir.target)
plt.xlabel(ir.feature_names[0])
plt.ylabel(ir.feature_names[1])
plt.title('Iris Dataset Visualization')
plt.show()