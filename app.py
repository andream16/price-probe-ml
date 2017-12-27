from statsmodels.tsa.arima_model import ARIMA

from configuration import configuration
from spark import spark
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from algorithms.arima.arima import difference

# Models
from model import price, currency

app_configuration = configuration.get_configuration('remote')
sc, url = spark.init_configuration(app_configuration)
spark.init_tables(sc, url)

plt.style.use('classic')

rdd_list_prices = price.get_prices(sc)
rdd_list_currencies = currency.get_currency(sc)

df_list_price = sc.createDataFrame(rdd_list_prices)
df_list_currencies = sc.createDataFrame(rdd_list_currencies)

pand_price = df_list_price.toPandas()
pand_curr = df_list_currencies.toPandas()

pand_curr = pand_curr.set_index('date')
pand_price = pand_price.set_index('date')

pand_merged = pand_price.merge(pand_curr, left_index=True, right_index=True)

pand_price.plot()
# load dataset
# series = Series.from_csv('daily-total-female-births.csv', header=0)
series = pand_merged.filter(items=['price', 'date', 'value'])
# evaluate parameters
p_values = [0, 1, 2, 4, 6, 8, 10]
d_values = range(0, 3)
q_values = range(0, 3)
warnings.filterwarnings("ignore")
# try all p,d,q values from lists and take only the best one
# evaluate_models(series.values, p_values, d_values, q_values)
X = series.head(len(series) - 10)['price']
Y = series.head(len(series) - 10)['value']
Z = series.tail(10)['value']
X.plot()
differenced = difference(X)
# fit model
model = ARIMA(X, order=(8, 0, 0), exog=Y)
model_fit = model.fit(disp=0)
# one-step out-of sample forecast
forecast = model_fit.forecast(exog=Z, steps=10)[0]
date_forecast = series.index._data[len(series.index._data) - 10: len(series.index._data)]
date_from = date_forecast[0]
date_to = date_forecast[9]
ser_forecast = pd.Series.from_array(forecast)
# invert the differenced forecast to something usable
#forecast = inverse_difference(X, forecast, 365)
print(forecast)
plt.plot(date_forecast, forecast)
plt.show()
