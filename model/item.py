from collections import namedtuple
from typing import List

# 3rd party
from pyspark import SparkContext

# Models
from model import category, price, review, trend

# DB
sql_entry = namedtuple("Category_Item", "item manufacturer has_reviews")
items = []


class ItemEntry:

    def __init__(self, item: str, manufacturer: str, has_reviews: bool):
        self.item = item
        self.manufacturer = manufacturer
        self.has_reviews = has_reviews


class Item:

    def __init__(self, item: str, manufacturer: str,
                 prices: List[price.PriceEntry], reviews: List[review.ReviewEntry],
                 trend_entries: List[trend.TrendEntry]):
        self.item = item
        self.manufacturer = manufacturer
        self.prices = prices
        self.reviews = reviews
        self.trend = trend_entries


# Helpers
def get_items(sc: SparkContext, page: int, size: int) -> List[ItemEntry]:
    start, end = 0, 0
    if page == 1:
        start = 1
        end = size
    else:
        start = ((page - 1) * size) + 1
        end = page * size
    #data_frame = sc.sql('SELECT item, manufacturer, has_reviews FROM item WHERE id BETWEEN "{}" AND "{}" ORDER BY id asc'
     #                   .format(start, end))

    data_frame = sc.sql(
        'SELECT item, manufacturer, has_reviews FROM item where id > 1 ORDER BY id asc')
    parsed_items: List[ItemEntry] = data_frame.rdd.map(lambda row: sql_entry(row[0], row[1], row[2])).collect()
    #if len(parsed_items) == size:
    for item in parsed_items:
        items.append(item)
        #page += 1
        #get_items(sc, page, size)
    return items
