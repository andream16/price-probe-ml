from collections import namedtuple
from typing import List

# 3rd party
from pyspark import SparkContext

# DB
sql_entry = namedtuple("Currency", "date value")
# Dictionary
currencies = {}


class CurrencyEntry:

    def __init__(self, date: str, value: float):
        self.date = date
        self.value = value


# Setters
def set_currency_by_key_and_entries(key: str, entries: List[CurrencyEntry]):
    if len(entries) > 0:
        for entry in entries:
            if key not in currencies:
                currencies[key] = [entry]
            else:
                currencies[key].append(entry)


# Getters
def get_currencies(sc: SparkContext):
    gpb_dollar_data_frame = get_currency_entries_by_key(sc, 'DOLLAR')
    gpb_dollar_parsed_entries = gpb_dollar_data_frame.rdd.map(
        lambda row: sql_entry(row[0], row[1])).collect()
    set_currency_by_key_and_entries('dollar', gpb_dollar_parsed_entries)
    gpb_euro_data_frame = get_currency_entries_by_key(sc, 'EURO')
    gpb_euro_parsed_entries = gpb_euro_data_frame.rdd.map(
        lambda row: sql_entry(row[0], row[1])).collect()
    set_currency_by_key_and_entries('euro', gpb_euro_parsed_entries)
    return currencies


# Helpers
def get_currency_entries_by_key(sc: SparkContext, key: str):
    return sc.sql('SELECT date, value FROM currency WHERE name = "{}"'.format(key))
