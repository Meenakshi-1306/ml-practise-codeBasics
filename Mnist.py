import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# Load the digits dataset
digits = load_digits()
print(digits.data)

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(digits.data, digits.target, test_size=0.2)

# Print the training data
print(X_train)
log=linear_model.LogisticRegression(max_iter=10000)
log.fit(X_train,Y_train)
src=log.score(X_test,Y_test)
ans=log.predict([digits.data[64]])
print(src)
print(ans)
plt.matshow(digits.images[64])
plt.show()