from collections import namedtuple

from dateutil import parser
from pyspark import SparkContext
from typing import List

Currencies = namedtuple("Currency", "name date value")


def set_currency_table(sc: SparkContext, url: str):

    df = sc.read.format("jdbc").options(
        url=url,
        driver='org.postgresql.Driver',
        dbtable='currency',
        user='postgres'
    ).load()
    df.createOrReplaceTempView("currency")


def get_currency(sc: SparkContext) -> List :
    df = sc.sql("Select name, date, value from currency where name = 'DOLLAR'")
    currencies = []
    curr = df.rdd.map(lambda row: Currencies(row[0], parser.parse(row[1]), row[2]))
    for p in curr.collect():
        currencies.append(p)
    return currencies
