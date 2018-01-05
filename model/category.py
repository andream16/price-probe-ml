from collections import namedtuple
from typing import List

# 3rd Party
from pyspark import SparkContext


# DB
sql_entry = namedtuple("Category_Item", "item category")
# Dictionary
categories_by_item = {}
items_by_category = {}


class CategoryEntry:

    def __init__(self, category):
        self.category = category


class Category:

    def __init__(self, item: str, categories: List[CategoryEntry]):
        self.item: str = item
        self.categories: List[CategoryEntry] = categories

    # Getters
    def get_category(self):
        return self

    @staticmethod
    def get_all_categories_by_item() -> dict:
        return categories_by_item

    @staticmethod
    def get_all_items_by_category() -> dict:
        return items_by_category


# Setters
def set_all_categories_by_item(item: str, categories: List[CategoryEntry]):
    if item not in categories_by_item:
        categories_by_item[item] = categories


def set_all_items_by_category(category: CategoryEntry, item: str):
    if category not in items_by_category:
        items_by_category[category] = [item]
    else:
        items_by_category[category].append(item)


# Helpers
def get_categories_by_item(item_id: str, sc: SparkContext):
    data_frame = sc.sql('SELECT item, category FROM category_item WHERE item = "{}"'.format(item_id))
    categories: List[CategoryEntry] = data_frame.rdd.map(lambda row: sql_entry(row[0], row[1])).collect()
    if len(categories) > 0:
        set_all_categories_by_item(item_id, categories)
    return categories
