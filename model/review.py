from collections import namedtuple
from typing import List

# 3rd party
from pyspark import SparkContext

# DB
sql_entry = namedtuple("review", "item date sentiment")


class ReviewEntry:

    def __init__(self, item: str, date: str, sentiment: int):
        self.item = item
        self.date = date
        self.sentiment = sentiment


class Review:

    # Helpers
    @staticmethod
    def get_reviews_by_item(sc: SparkContext, item: str):
        data_frame = sc.sql('SELECT item, date, sentiment FROM review WHERE item = "{}"'.format(item))
        parsed_reviews: List[ReviewEntry] = data_frame.rdd.map(
            lambda row: sql_entry(row[0], row[1], row[2])).collect()
        if len(parsed_reviews) > 0:
            return parsed_reviews
