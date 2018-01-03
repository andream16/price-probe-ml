from collections import namedtuple
from typing import List

# 3rd Party
from pyspark import SparkContext

# This class models a Category database entry. It also has two variables to get all items by a category and vice-versa.


class Category:

    # DB
    sql_entry = namedtuple("Category_Item", "category")
    # Dictionary
    categories_by_item = {}
    items_by_category = {}

    def __init__(self, item: str, categories: List[str]):
        self.item: str = item
        self.categories: List[str] = categories

    # Getters
    def get_category(self):
        return self

    def get_all_categories_by_item(self) -> dict:
        return self.categories_by_item

    def get_all_items_by_category(self) -> dict:
        return self.items_by_category

    # Setters
    def set_all_categories_by_item(self, item: str, categories: List[str]):
        if item not in self.categories_by_item:
            self.categories_by_item[item] = categories

    def set_all_items_by_category(self, category: str, item: str):
        if category not in self.items_by_category:
            self.items_by_category[category] = [item]
        else:
            self.items_by_category[category].append(item)

    # Helpers
    def get_categories_by_item(self, item_id: str, sc: SparkContext):
        data_frame = sc.sql('SELECT category FROM category_item WHERE item = "{}"'.format(item_id))
        categories = data_frame.rdd.map(lambda row: self.sql_entry(row[0])).collect()
        for c in categories:
            if c:
                self.categories.append(c)
                self.set_all_items_by_category(c, item_id)
        if len(self.categories) == 0:
            return ValueError("No Categories found for item = %s", item_id)
        self.item = item_id
        self.set_all_categories_by_item(item_id, categories)
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
