from collections import namedtuple

import numpy
from pyspark.sql.types import StructType, StructField, StringType, FloatType, BooleanType
from statsmodels.tsa.arima_model import ARIMA

from configuration import configuration
from spark import spark
from typing import List
import pandas as pd
from dateutil import parser
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings
import statsmodels.api as sm
import itertools
from arima import evaluate_models, difference, inverse_difference

app_configuration = configuration.get_configuration('remote')
sc, url = spark.init_configuration(app_configuration)
spark.init_tables(sc, url)

plt.style.use('classic')


Prices = namedtuple("Price", "item date price flag")
Currencies = namedtuple("Currency", "name date value")


def get_prices() -> List:
    df = sc.sql("Select p.item, p.date, p.price, p.flag from price p where p.item ='B00DP4AR10'")
    prices = []
    prz = df.rdd.map(lambda row: Prices(row[0],  parser.parse(row[1]), row[2], row[3]))
    for p in prz.collect():
        prices.append(p)
    #print(len(prices))
    return prices


def get_currency() -> List :
    df = sc.sql("Select name, date, value from currency where name = 'DOLLAR'")
    currencies = []
    curr = df.rdd.map(lambda row: Currencies(row[0], parser.parse(row[1]), row[2]))
    for p in curr.collect():
        currencies.append(p)
    # print(len(prices))
    return currencies


rdd_list_prices = get_prices()
rdd_list_currencies = get_currency()

df_list_price = sc.createDataFrame(rdd_list_prices)
df_lost_currencies = sc.createDataFrame(rdd_list_currencies)

pand_price = df_list_price.toPandas()
pand_curr = df_lost_currencies.toPandas()
#pand_price.append(pand_curr)
pand_curr = pand_curr.set_index('date')
pand_price = pand_price.set_index('date')

pand_merged = pand_price.merge(pand_curr,left_index=True,right_index=True)

pand_price.plot()
# load dataset
#series = Series.from_csv('daily-total-female-births.csv', header=0)
series = pand_merged.filter(items=['price','date','value'])
# evaluate parameters
p_values = [0, 1, 2, 4, 6, 8, 10]
d_values = range(0, 3)
q_values = range(0, 3)
warnings.filterwarnings("ignore")
#try all p,d,q values from lists and take only the best one
#evaluate_models(series.values, p_values, d_values, q_values)
X = series.head(len(series) - 10)['price']
Y = series.head(len(series) - 10)['value']
Z = series.tail(10)['value']
X.plot()
differenced = difference(X)
# fit model
model = ARIMA(X, order=(8,0,0),exog=Y)
model_fit = model.fit(disp=0)
# one-step out-of sample forecast
forecast = model_fit.forecast(exog=Z,steps=10)[0]
date_forecast = series.index._data[len(series.index._data) - 10: len(series.index._data)]
date_from = date_forecast[0]
date_to = date_forecast[9]
ser_forecast = pd.Series.from_array(forecast)
# invert the differenced forecast to something usable
#forecast = inverse_difference(X, forecast, 365)
print(forecast)
plt.plot(date_forecast,forecast)
plt.show()