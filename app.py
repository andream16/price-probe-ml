from collections import namedtuple

from pyspark.sql.types import StructType, StructField, StringType, FloatType, BooleanType

from configuration import configuration
from spark import spark
from typing import List


app_configuration = configuration.get_configuration('remote')
sc, url = spark.init_configuration(app_configuration)
spark.init_tables(sc, url)


Prices = namedtuple("Price", "item date price flag")


def get_prices() -> List:
    df = sc.sql("Select p.item, p.date, p.price, p.flag from price p")
    prices = []
    prz = df.rdd.map(lambda row: Prices(row[0], row[1], row[2], row[3]))
    for p in prz.collect():
        prices.append(p)
    print(len(prices))
    return prices


rdd_list = get_prices()

df_list = sc.createDataFrame(rdd_list)
df_list.show(100)
