from collections import namedtuple
from typing import List

# 3rd Party
from pyspark import SparkContext


class Category:

    # DB
    sqlEntry = namedtuple("Category_Item", "category")
    # Dictionary
    categories_by_item = {}

    def __init__(self, item: str, categories: List[str]):
        self.item: str = item
        self.categories: List[str] = categories

    def get_category(self):
        return self

    def get_all_categories_by_item(self):
        return self.categories_by_item

    def get_categories_by_item(self, item_id: str, sc: SparkContext):
        query = 'SELECT category FROM category_item WHERE item = "{}"'.format(item_id)
        df = sc.sql(query)
        try:
            raw_categories = df.rdd.map(lambda row: self.sqlEntry(row[0]))
        except IndexError:
            return ValueError("No Categories found for item = %s", item_id)
        categories = raw_categories.collect()
        for c in categories:
            if c:
                self.categories.append(c)
        if len(self.categories) == 0:
            return ValueError("No Categories found for item = %s", item_id)
        self.item = item_id
        if item_id not in self.categories_by_item:
            self.categories_by_item[item_id] = categories
        return self

    @staticmethod
    def set_category_item_table(sc: SparkContext, url: str):
        df = sc.read.format("jdbc").options(
            url=url,
            driver='org.postgresql.Driver',
            dbtable='category_item',
            user='postgres'
        ).load()
        df.createOrReplaceTempView("category_item")
