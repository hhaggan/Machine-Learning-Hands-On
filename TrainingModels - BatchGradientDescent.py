import numpy as np 
import pandas as pd 

#generating random data
x = 2*np.random.rand(100,1)
y = 4 + 3 * x + np.random.randn(100,1)

x_b = np.c_[np.ones((100,1)), x]
theta_best = np.linalg.inv(x_b.T.dot(x_b)).dot(x_b.T).dot(y)

x_new = np.array([[0], [2]])
x_new_b = np.c_[np.ones((2,1)), x_new]

y_predict = x_new_b.dot(theta_best)

#Batch Gradient Descent Start

eta = 0.1
n_iterations = 1000
m = 100

theta = np.random.randn(2,1)

for iteration in range(n_iterations):
    gradients = 2/m * x_b.T.dot(x_b.dot(theta)- y)
    theta = theta - eta * gradients