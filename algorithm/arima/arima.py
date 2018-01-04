
import numpy
# 3rd party
from pyspark import SparkContext
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf, pacf
# Models
from model import price, currency


# evaluate an ARIMA model for a given order (p,d,q)
def evaluate_arima_model(X, arima_order):
    # prepare training dataset
    train_size = int(len(X) * 0.66)
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]
    # make predictions
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=arima_order)
        model_fit = model.fit(disp=0)
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])
    # calculate out of sample error
    error = mean_squared_error(test, predictions)
    return error


# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values):
    dataset = dataset.astype('float32')
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p, d, q)
                try:
                    mse = evaluate_arima_model(dataset, order)
                    if mse < best_score:
                        best_score, best_cfg = mse, order
                    print('ARIMA%s MSE=%.3f' % (order, mse))
                except:
                    continue
    print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))
    return best_cfg


# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return numpy.array(diff)


# invert differenced value
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]


def test_arima(sc: SparkContext):
    plt.style.use('classic')

    rdd_list_prices = price.get_prices(sc)
    rdd_list_currencies = currency.get_currency(sc)

    df_list_price = sc.createDataFrame(rdd_list_prices)
    df_list_currencies = sc.createDataFrame(rdd_list_currencies)

    pand_price = df_list_price.toPandas()
    pand_curr = df_list_currencies.toPandas()

    pand_curr = pand_curr.set_index('date')
    pand_price = pand_price.set_index('date')
    print(sm.tsa.stattools.adfuller(pand_price['price']))
    print(sm.tsa.stattools.adfuller(pand_curr['value']))
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(pand_price['price'], lags=40, ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(pand_price['price'], lags=40, ax=ax2)
    plt.show()

    lag_acf = acf(pand_price['price'], nlags=40)
    lag_pacf = pacf(pand_price['price'], nlags=40)

    print('acf', lag_acf)
    print('pacf', lag_pacf)
    # threshold values for acf and pacf
    positive_thrshld = 1.96 / np.sqrt(len(pand_price['price']))
    negative_thrshld = -1.96 / np.sqrt(len(pand_price['price']))

    p = 0
    q = 0
    for value in lag_acf:
        if value < positive_thrshld:
            q = lag_acf.index(value)
            break
    for value in lag_pacf:
        if value < positive_thrshld:
            p = lag_pacf.index(value)
            break
    print('p', p)
    print('q', q)
    pand_merged = pand_price.merge(pand_curr, left_index=True, right_index=True)

    pand_price.plot()
    # load dataset
    # series = Series.from_csv('daily-total-female-births.csv', header=0)
    series = pand_merged.filter(items=['price', 'date', 'value'])
    # evaluate parameters
    p_values = [0, 6, 7, 8, 10]
    d_values = range(0, 3)
    q_values = range(0, 3)
    warnings.filterwarnings("ignore")

    # try all p,d,q values from lists and take only the best one
    # best_order = evaluate_models(series['price'], p_values, d_values, q_values)
    best_order = [7, 1, 0]
    size = int(len(series) * 0.66)
    X, test = series[0:size]['price'], series[size:]['price']
    Y = series.head(len(series) - 10)['value']
    Z = series.tail(10)['value']
    X.plot()
    # differenced = difference(X)

    # fit model
    model = ARIMA(X, order=best_order)
    model_fit = model.fit(disp=0)
    # one-step out-of sample forecast
    forecast = model_fit.forecast(steps=len(test))[0]
    forecast_ewma = pd.ewma(X, span=7)
    print(forecast_ewma)
    date_forecast = test.index._data
    print(len(date_forecast))
    print(len(test))
    # date_from = date_forecast[0]
    # date_to = date_forecast[9]
    # ser_forecast = pd.Series.from_array(forecast)

    # invert the differenced forecast to something usable
    # forecast = inverse_difference(X, forecast, 365)
    print(forecast)
    plt.plot(date_forecast, forecast)
    plt.show()
