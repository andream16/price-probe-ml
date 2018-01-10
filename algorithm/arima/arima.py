
# 3rd party
import numpy
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import acf, pacf, adfuller

results = {}


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
                except:
                    continue
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


def find_d(series, threshold, d):
    if d > 0:
        d += 1
    adfuller_test_results = adfuller(series.values)
    if (adfuller_test_results[0] < 0 and adfuller_test_results[0] < threshold) or d > 10:
        return d
    series = series.diff()
    find_d(series, threshold, d)


def test_arima(title, data_frame_dict):

    data_frame_dict['data_frame']['date'] = pd.to_datetime(data_frame_dict['data_frame']['date'])

    # Setting index on date
    data_frame = data_frame_dict['data_frame'].set_index('date')

    prices_column = data_frame['price']
    prices_elements_number = len(prices_column)

    # ACF = Autocorrelation Function. How big are clusters of data containing elements with similar trend.
    # https://onlinecourses.science.psu.edu/stat510/node/60
    lag_acf = acf(prices_column, nlags=prices_elements_number)
    # PACF = Partially Autocorrelation Function. Correlation between points not looking at already visited ones.
    # https://onlinecourses.science.psu.edu/stat510/node/46
    lag_pacf = pacf(prices_column, nlags=prices_elements_number)

    # Threshold values for ACF and PACF for 95% Confidence Interval
    positive_threshold = 1.96 / np.sqrt(prices_elements_number)

    # First time a ACF value crosses positive threshold (AR)
    p = 0
    # Number of times needed to make the series stationary (I)
    d = find_d(prices_column, 0.05, 0)
    # First time a PACF value crosses positive threshold (MA)
    q = 0
    acf_it = 0
    pacf_it = 0
    for value in np.nditer(lag_acf):
        if value < positive_threshold:
            q = acf_it
            break
        else:
            acf_it += 1
    for value in np.nditer(lag_pacf):
        if value < positive_threshold:
            p = pacf_it
            break
        else:
            pacf_it += 1

    best_configuration = evaluate_models(prices_column, range(0, p), [d], range(0, q))

    size = int(len(prices_column) * 0.66)
    training_set, test = prices_column[0:size], prices_column[size:]

    # External Columns
    column_names = title.split(',')
    if len(column_names) > 3:
        selected_columns_names = [k for k in column_names if (k not in ['price', 'item', 'date', 'category', 'manufacturer'])]
        selected_columns = data_frame[selected_columns_names]
        selected_columns, selected_columns_test = selected_columns[0:size], selected_columns[size:]
        if len(selected_columns_names) > 0:
            model = ARIMA(training_set, order=best_configuration, exog=selected_columns)
            model_fit = model.fit(disp=0)
            forecast = model_fit.forecast(steps=len(test), exog=selected_columns_test)[0]
        else:
            return
    else:
        # Fit model
        model = ARIMA(training_set, order=best_configuration)
        model_fit = model.fit(disp=0)
        # one-step out-of sample forecast
        forecast = model_fit.forecast(steps=len(test))[0]
    date_forecast = test.index._data

    results[title] = {
        'forecast': forecast, 'date_forecast': date_forecast,
        'score': mean_squared_error(test, forecast), 'prices': prices_column,
        'training_set': training_set
    }


def plot_best_result():
    best_score = 0

    best_result = {}
    for attr, value in results.items():
        if best_score == 0 or results[attr]['score'] < best_score:
            best_score = results[attr]['score']
            best_result = {
                'name': attr, 
                'prices': results[attr]['prices'],
                'training_set': results[attr]['training_set'],
                'date_forecast': results[attr]['date_forecast'],
                'forecast': results[attr]['forecast']
            }
    plt.style.use('classic')
    best_result['prices'].plot()
    best_result['training_set'].plot()
    plt.plot(best_result['date_forecast'], best_result['forecast'])
    plt.title(best_result['name'])
    plt.show()
    return best_result
