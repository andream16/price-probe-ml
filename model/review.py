from collections import namedtuple
from typing import List

# 3rd party
from pyspark import SparkContext

# DB
sql_entry = namedtuple("review", "item date sentiment stars")


class ReviewEntry:

    def __init__(self, item: str, date: str, sentiment: int, stars):
        self.item = item
        self.date = date
        self.sentiment = sentiment
        self.stars = stars


class Review:

    # Helpers
    @staticmethod
    def get_reviews_by_item(sc: SparkContext, item: str):
        data_frame = sc.sql('SELECT item, date, sentiment, stars FROM review WHERE item = "{}"'.format(item))
        parsed_reviews: List[ReviewEntry] = data_frame.rdd.map(
            lambda row: sql_entry(row[0], row[1], row[2], row[3])).collect()
        if len(parsed_reviews) > 0:
            map(lambda entry: float("{0:.2f}".format(entry.sentiment)), parsed_reviews)
            map(lambda entry: float("{0:.2f}".format(entry.stars)), parsed_reviews)
            return parsed_reviews
