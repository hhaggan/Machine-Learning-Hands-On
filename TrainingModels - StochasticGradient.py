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

#stochastic Gradient Descent Start

n_epcoh = 50
t0, t1 = 5, 50
#eta = 0.1
n_iterations = 1000
m = 100


def learning_schedule(t):
    return t0 / (t+t1)

theta = np.random.randn(2,1)

for epoch in range(n_epcoh):
    for i in range(m):
        random_index = np.random.randint(m)
        xi = x_b[random_index:random_index+1]
        yi = y[random_index:random_index+1]
        gradients = 2*xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule(epoch*m+i)
        theta = theta - eta * gradients

from sklearn.linear_model import SGDRegressor
sgd_reg = SGDRegressor(max_iter=50, penalty=None,eta0=0.1)
sgd_reg.fit(x, y.ravel())

sgd_reg.intercept_, sgd_reg.coef_