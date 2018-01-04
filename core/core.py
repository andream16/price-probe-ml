from typing import List

# 3rd party
from pyspark import SparkContext
import pandas as pd
from itertools import product

# Models
from model import category, currency, item, manufacturer, price, review, trend

NO_MANUFACTURER = 'no_manufacturer'

def start_algorithm(sc: SparkContext):
    items = get_parsed_items(sc)
    feature_dict_creator(items[0])


def get_parsed_items(sc: SparkContext) -> List[item.Item]:

    currencies = currency.get_currencies(sc)
    items = item.get_items(sc, 1, 10)
    final_items = []
    if len(items) > 0:
        for i in items:
            current_item_id = i.item
            current_manufacturer = i.manufacturer
            current_trend = []
            if current_manufacturer:
                manufacturer.Manufacturer.add_item_by_manufacturer_entry(current_manufacturer, current_item_id)
                manufacturer.Manufacturer.add_manufacturer_by_item_entry(current_manufacturer, current_item_id)
                if current_manufacturer != NO_MANUFACTURER:
                    current_trend = trend.get_trend_by_manufacturer(sc, current_manufacturer)
            current_prices = price.get_prices_by_item(sc, current_item_id)
            current_reviews = []
            if i.has_reviews:
                current_reviews = review.Review.get_reviews_by_item(sc, current_item_id)
            current_categories = category.get_categories_by_item(current_item_id, sc)
            final_items.append(item.Item(current_item_id, current_categories, current_manufacturer, current_prices,
                                         current_reviews, current_trend))
    return final_items


def feature_dict_creator(itm: item.Item):
    data_frames_combinations, data_frames_all_features_dict = ({},)*2
    for attr, value in itm.__dict__.items():
        if attr not in ['item', 'manufacturer'] and len(value) > 0:
            data_frames_all_features_dict[attr] = get_pandas_data_frame_from_list({attr: value})
        if attr in ['item', 'manufacturer'] and value != NO_MANUFACTURER:
            data_frames_all_features_dict[attr] = get_pandas_data_frame_from_list({attr: [value]})
    # All features dictionary
    all_features_key = get_key_name_by_object(data_frames_all_features_dict)
    data_frames_combinations[all_features_key] = data_frames_all_features_dict
    return data_frames_combinations


def join_data_frames_by_optional_key(**kwargs) -> pd.DataFrame or None:
    key = kwargs.get('key', None)
    df1 = kwargs.get('df1', None)
    df2 = kwargs.get('df2', None)
    if not df1 or not df2:
        return
    if not key:
        return product(df1, df2)
    return df1.merge(df2, key)


def get_key_name_by_object(obj: any) -> str:
    final_key_name = ''
    for key in obj.items():
        final_key_name += key + ','
        if len(obj[key].items()) > 2:
            for inner_key in obj[key].items():
                final_key_name += inner_key + ','
    return final_key_name


def get_pandas_data_frame_from_list(generic_entry: any) -> pd.DataFrame:
    return pd.DataFrame(data=generic_entry)
