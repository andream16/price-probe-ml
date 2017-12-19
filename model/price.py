from collections import namedtuple

from dateutil import parser
from pyspark import SparkContext
from typing import List

Prices = namedtuple("Price", "item date price flag")


def set_price_table(sc: SparkContext, url: str):
    df = sc.read.format("jdbc").options(
        url=url,
        driver='org.postgresql.Driver',
        dbtable='price',
        user='postgres'
    ).load()
    df.createOrReplaceTempView("price")


def get_prices(sc: SparkContext) -> List:
    df = sc.sql("Select p.item, p.date, p.price, p.flag from price p where p.item ='B00DP4AR10'")
    prices = []
    prz = df.rdd.map(lambda row: Prices(row[0], parser.parse(row[1]), row[2], row[3]))
    for p in prz.collect():
        prices.append(p)
    return prices
