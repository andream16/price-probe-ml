from collections import namedtuple
from dateutil import parser
from typing import List

# 3rd party
from pyspark import SparkContext


class CurrencyEntry:

    def __init__(self, date: str, value: float):
        self.date = date
        self.value = value


class Currency:

    # DB
    sql_entry = namedtuple("Currency", "name date value")
    # Dictionary
    currencies = {}

    def __init__(self, name: str, entries: List[CurrencyEntry]):
        self.name: str = name
        self.entries: List[CurrencyEntry] = entries

    # Setters
    def set_currency_by_key_and_entries(self, key: str, entries: List[CurrencyEntry]):
        if len(entries) > 0:
            for entry in entries:
                if key not in self.currencies:
                    self.currencies[key] = [entry]
                else:
                    self.currencies[key].append(entry)

    # Getters
    def get_currencies(self, sc: SparkContext):
        gpb_dollar_data_frame = self.get_currency_entries_by_key(sc, 'DOLLAR')
        gpb_dollar_parsed_entries = gpb_dollar_data_frame.rdd.map(
            lambda row: self.sql_entry(row[0], parser.parse(row[1]), row[2])).collect()
        self.set_currency_by_key_and_entries('dollar', gpb_dollar_parsed_entries)
        gpb_euro_data_frame = self.get_currency_entries_by_key(sc, 'EURO')
        gpb_euro_parsed_entries = gpb_euro_data_frame.rdd.map(
            lambda row: self.sql_entry(row[0], parser.parse(row[1]), row[2])).collect()
        self.set_currency_by_key_and_entries('euro', gpb_euro_parsed_entries)
        return self

    # Helpers
    @staticmethod
    def get_currency_entries_by_key(sc: SparkContext, key: str):
        return sc.sql('SELECT name, date, value FROM currency WHERE name = "{}"'.format(key))

    @staticmethod
    def set_currency_table(sc: SparkContext, url: str):
        df = sc.read.format("jdbc").options(
            url=url,
            driver='org.postgresql.Driver',
            dbtable='currency',
            user='postgres'
        ).load()
        df.createOrReplaceTempView("currency")
