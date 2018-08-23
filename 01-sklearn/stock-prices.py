import pandas as pd
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt


def get_data(filename):
    df = pd.read_csv(filename)

    dates = df['Date'].values
    prices = df['Close'].values

    return dates, prices


def predict_prices(dates, prices, x):
    dates = np.reshape(dates, (len(dates), 1))

    svr_len = SVR(kernel='linear', C=1e3)
    svr_poli = SVR(kernel='poly', C=1e3, degree=2)
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)

    svr_len.fit(dates, prices)
    svr_poli.fit(dates, prices)
    svr_rbf.fit(dates, prices)

    plt.scatter(dates, prices, color='black', label='Data')
    plt.plot(dates, svr_rbf.predict(dates), color='red', label='RBF')
    plt.plot(dates, svr_len.predict(dates), color='green', label='Linear')
    plt.plot(dates, svr_poli.predict(dates), color='blue', label='Poly')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Support Vector Regression')
    plt.legend()
    plt.show()

    return svr_len.predict(x)[0], svr_poli.predict(x)[0], svr_rbf.predict(x)[0]


dates, prices = get_data('aapl.csv')


# prices = predict_prices(dates, prices, 29)
# print(prices)
