from collections import namedtuple
from typing import List

# 3rd party
from pyspark import SparkContext

# DB
sql_entry = namedtuple("Price", "item, date price")


class PriceEntry:

    def __init__(self, item: str, date: str, price: float):
        self.item = item
        self.date = date
        self.price = price


# Helpers
def get_prices_by_item(sc: SparkContext, item: str):
    data_frame = sc.sql('SELECT item, date, price FROM price WHERE item = "{}"'.format(item))
    parsed_prices: List[PriceEntry] = data_frame.rdd.map(
        lambda row: sql_entry(row[0], row[1], row[2])).collect()
    if len(parsed_prices) > 0:
        map(lambda price_entry: float("{0:.2f}".format(price_entry.price)), parsed_prices)
        return parsed_prices
