
# 3rd party
import numpy
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMA
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf, pacf, adfuller

results = {}


# evaluate an ARIMA model for a given order (p,d,q)
def evaluate_arima_model(X, arima_order):
    # prepare training dataset
    train_size = int(len(X) * 0.9)
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]
    # make predictions
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=arima_order)
        model_fit = model.fit(disp=-1)
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])
    # calculate out of sample error
    error = mean_squared_error(test, predictions)
    return error


# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values):
    dataset = dataset.astype('float32')
    best_score, best_cfg = float("inf"), (0, 0, 0)
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
    adfuller_test_results = adfuller(series.values)
    if (adfuller_test_results[0] < 0 and adfuller_test_results[4]['5%'] > adfuller_test_results[0] and adfuller_test_results[0] < threshold) or d > 2:
        return d
    else:
        series = series.diff()
        series = series.dropna()
        d += 1
        return find_d(series, threshold, d)


def test_arima(title, dict):

    prices_column = dict['prices_column']
    data_frame = dict['data_frame']
    p, d, q = dict['order']
    data_frame['date'] = pd.to_datetime(data_frame['date'])

    # Setting index on date
    data_frame = data_frame.set_index('date')
    best_configuration = evaluate_models(prices_column, range(0, p), [d], range(0, q))
    size = int(len(prices_column) * 0.9)
    training_set, test = prices_column[0:size], prices_column[size:]

    # External Columns
    column_names = title.split(',')
    print(best_configuration)
    print(title)
    if len(column_names) > 3:
        selected_columns_names = [k for k in column_names if (k not in ['price', 'item', 'date', 'manufacturer'])]
        selected_columns = data_frame[selected_columns_names]
        selected_columns, selected_columns_test = selected_columns[0:size], selected_columns[size:]
        flag_flag = False
        flag_trend = False
        count_trend = 0
        count_flag = 0
        check_duplicated_flag = pd.Series()
        check_duplicated_trend = pd.Series()
        if len(selected_columns_names) > 0:
            if 'flag' in selected_columns.columns:
                check_duplicated_flag = selected_columns.duplicated(subset='flag', keep=False)
            if 'trend_value' in selected_columns.columns:
                check_duplicated_trend = selected_columns.duplicated(subset='trend_value', keep=False)

            if check_duplicated_flag.size > 0:
                for value in np.nditer(check_duplicated_flag.values):
                    if value:
                        count_flag += 1
            if check_duplicated_trend.size > 0:
                for value in np.nditer(check_duplicated_trend.values):
                    if value:
                        count_trend += 1
            if count_trend == check_duplicated_trend.size:
                flag_trend = True
            if count_flag == check_duplicated_flag.size:
                flag_flag = True
            if not flag_flag and not flag_trend:
                model = ARIMA(training_set, order=best_configuration, exog=selected_columns)
                model_fit = model.fit(disp=-1)
                forecast = model_fit.forecast(steps=len(test), exog=selected_columns_test)[0]
            elif not flag_trend:
                model = ARIMA(training_set, order=best_configuration, exog=selected_columns['trend_value'])
                model_fit = model.fit(disp=-1)
                forecast = model_fit.forecast(steps=len(test), exog=selected_columns_test['trend_value'])[0]
            elif not flag_flag:
                model = ARIMA(training_set, order=best_configuration, exog=selected_columns['flag'])
                model_fit = model.fit(disp=-1)
                forecast = model_fit.forecast(steps=len(test), exog=selected_columns_test['flag'])[0]
            else:
                model = ARIMA(training_set, order=best_configuration)
                model_fit = model.fit(disp=-1)
                forecast = model_fit.forecast(steps=len(test))[0]
        else:
            return
    else:
        # Fit model
        model = ARIMA(training_set, order=best_configuration)
        model_fit = model.fit(disp=-1)
        # one-step out-of sample forecast
        forecast = model_fit.forecast(steps=len(test))[0]
    date_forecast = test.index._data

    results[title] = {
        'forecast': forecast, 'date_forecast': date_forecast,
        'score': mean_absolute_percentage_error(test, forecast), 'prices': prices_column,
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
                'forecast': results[attr]['forecast'],
                'score': best_score
            }
    plt.style.use('classic')
    best_result['prices'].plot()
    best_result['training_set'].plot()
    plt.plot(best_result['date_forecast'], best_result['forecast'])
    plt.title(best_result['name'])
    plt.show()
    return best_result


def find_p_d_q_values(prices_column:pd.Series, prices_elements_number:int):

    # First time a ACF value crosses positive threshold (AR)
    p = 0
    # Number of times needed to make the series stationary (I)
    d = find_d(prices_column, 0.05, 0)
    # First time a PACF value crosses positive threshold (MA)
    q = 0

    p, q = compute_acf_pacf(prices_column, prices_elements_number)

    return p, d, q


def compute_acf_pacf(prices_column: pd.Series, prices_elements_number:int):
    # ACF = Autocorrelation Function. How big are clusters of data containing elements with similar trend.
    # https://onlinecourses.science.psu.edu/stat510/node/60
    # Threshold values for ACF and PACF for 95% Confidence Interval
    positive_threshold = 1.96 / np.sqrt(prices_elements_number)
    if prices_column.min() == prices_column.max():
        return 0, 0
    lag_acf = acf(prices_column, nlags=prices_elements_number)
    # PACF = Partially Autocorrelation Function. Correlation between points not looking at already visited ones.
    # https://onlinecourses.science.psu.edu/stat510/node/46
    lag_pacf = pacf(prices_column, nlags=prices_elements_number)
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(prices_column, lags=40, ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(prices_column, lags=40, ax=ax2)
    plt.show()
    acf_it = 0
    pacf_it = 0
    for value in np.nditer(lag_acf):
        if value < positive_threshold:
            q = acf_it
            if value < -positive_threshold:
                q += 1
            break
        else:
            acf_it += 1

    for value in np.nditer(lag_pacf):
        if value < positive_threshold:
            p = pacf_it
            if value < -positive_threshold:
                p += 1
            break
        else:
            pacf_it += 1
    return p, q


def find_arima_parameters_by_dataframe(data_frame):
    dict = {}
    data_frame['date'] = pd.to_datetime(data_frame['date'])

    # Setting index on date
    data_frame = data_frame.set_index('date')

    prices_column = data_frame['price']
    prices_elements_number = len(prices_column)

    p, d, q = find_p_d_q_values(prices_column, prices_elements_number)
    print(p, d, q)
    dict['order'] = p, d, q
    dict['prices_column'] = prices_column
    dict['data_frame'] = data_frame
    return dict


# https://en.wikipedia.org/wiki/Mean_absolute_percentage_error
def mean_absolute_percentage_error(test, forecast):
    m = (100 / len(test))
    abs_sum = 0.0
    for i in test:
        abs_sum += abs((test[i] - forecast[i]) / test[i])
    return abs_sum * m
