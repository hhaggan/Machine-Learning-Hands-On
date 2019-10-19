import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#df = pd.read_csv("")

x = 2*np.random.rand(100,1)
y = 4 + 3 * x + np.random.randn(100,1)

x_b = np.c_[np.ones((100,1)), x]
theta_best = np.linalg.inv(x_b.T.dot(x_b)).dot(x_b.T).dot(y)

x_new = np.array([[0], [2]])
x_new_b = np.c_[np.ones((2,1)), x_new]

y_predict = x_new_b.dot(theta_best)

plt.plot(x_new, y_predict, "r-")
plt.plot(x, y, "b.")
plt.axis([0,2,0,15])
plt.show()

from sklearn.linear_model import LinearRegression
linear_regression = LinearRegression()
linear_regression.fit(x, y)
linear_regression.intercept_, linear_regression.coef_

linear_regression.predict(x_new)
theta_best_svd, residuals, rank, s = np.linalg.lstsq(x_b, y, rcond=1e-6)

np.linalg.pinv(x_b).dot(y)