from collections import namedtuple
from typing import List

# 3rd party
from pyspark import SparkContext

# DB
sql_entry = namedtuple("review", "date value")


class TrendEntry:

    def __init__(self, date: str, value: float):
        self.date = date
        self.value = value


# Helpers
def get_trend_by_manufacturer(sc: SparkContext, manufacturer: str):
    data_frame = sc.sql('SELECT date, value FROM trend WHERE manufacturer = "{}"'.format(manufacturer))
    parsed_trend_entries: List[TrendEntry] = data_frame.rdd.map(
        lambda row: sql_entry(row[0], row[1])).collect()
    if len(parsed_trend_entries) > 0:
        return parsed_trend_entries
