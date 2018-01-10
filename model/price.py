from collections import namedtuple
from typing import List

# 3rd party
from pyspark import SparkContext

# DB
sql_entry = namedtuple("Price", "item date price flag")


class PriceEntry:

    def __init__(self, item: str, date: str, price: float, flag: float):
        self.item = item
        self.date = date
        self.price = price
        self.flag = flag


# Helpers
def get_prices_by_item(sc: SparkContext, item: str):
    data_frame = sc.sql('SELECT item, date, price, flag FROM price WHERE item = "{}"'.format(item))
    parsed_prices = data_frame.rdd.map(
        lambda row: sql_entry(row[0], row[1], row[2], row[3])).collect()
    if len(parsed_prices) > 0:
        map(lambda price_entry: float("{0:.2f}".format(price_entry.price)), parsed_prices)
        flagged_prices = []
        for i, p in enumerate(parsed_prices):
            if p.flag:
                flagged_prices.append(PriceEntry(p.item, p.date, p.price, 1000.0))
            else:
                flagged_prices.append(PriceEntry(p.item, p.date, p.price, 0.0))
        prices_tuple = []
        for price in flagged_prices:
            prices_tuple.append(sql_entry(price.item, price.date, price.price, price.flag))
        return prices_tuple
