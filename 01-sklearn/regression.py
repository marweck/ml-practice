import pandas as pd
import numpy as np
from sklearn import linear_model, tree
import matplotlib.pyplot as plt

# read data
dataframe = pd.read_fwf('brain_body.txt')
x_values = dataframe[['Brain']]
y_values = dataframe[['Body']]

regressor = linear_model.LinearRegression()
regressor.fit(x_values, y_values)

regressor2 = tree.DecisionTreeRegressor()
regressor2.fit(x_values, y_values)

plt.scatter(x_values, y_values)
plt.plot(x_values, regressor.predict(x_values), regressor2.predict(x_values))
plt.show()


class LogisticRegression:

    """
    This is class is just a simple implementation of 2-class classifier
    """

    def __init__(self, lr=0.01, num_iterations=100000, fit_intercept=True, verbose=False):
        self.lr = lr
        self.num_iterations = num_iterations
        self.fit_intercept = fit_intercept
        self.verbose = verbose

    def __add_intercept__(self, X):
        intercept = np.ones(X.shape[0], 1)
        return np.concatenate((intercept, X), axis=1)

    def __sigmoid__(self, z):
        return 1 / (1 + np.exp(-z))

    def __loss__(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

    def fit(self, X, y):
        if self.fit_intercept:
            X = self.__add_intercept__(X)

        self.theta = np.zeros(X.shape[1])

        for i in range(self.num_iterations):
            z = np.dot(X, self.theta)
            h = self.__sigmoid__(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta -= self.lr * gradient

            if self.verbose == True and i % 10000 == 0:
                z = np.dot(X, self.theta)
                h = self.__sigmoid__(z)
                print(f'loss: {self.__loss__(h, y)}')

    def predict_prob(self, X):
        if self.fit_intercept:
            X = self.__add_intercept__(X)

        return self.__sigmoid__(np.dot(X, self.theta))

    def predict(self, X, threshold=0.5):
        return self.predict_prob(X) >= threshold
