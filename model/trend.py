from collections import namedtuple
from typing import List

# 3rd party
from pyspark import SparkContext

# DB
sql_entry = namedtuple("trend", "manufacturer date trend_value")


class TrendEntry:

    def __init__(self, manufacturer: str, date: str, trend_value: float):
        self.manufacturer = manufacturer
        self.date = date
        self.trend_value = trend_value


# Helpers
def get_trend_by_manufacturer(sc: SparkContext, manufacturer: str):
    data_frame = sc.sql('SELECT manufacturer, date, value FROM trend WHERE manufacturer = "{}"'.format(manufacturer))
    parsed_trend_entries: List[TrendEntry] = data_frame.rdd.map(
        lambda row: sql_entry(row[0], row[1], row[2])).collect()
    map(lambda entry: float("{0:.2f}".format(entry.trend_value)), parsed_trend_entries)
    if len(parsed_trend_entries) > 0:
        return parsed_trend_entries
