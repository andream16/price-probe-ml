from typing import List
from itertools import combinations, chain

# 3rd party
from pyspark import SparkContext
import pandas as pd

# ARIMA
from algorithm.arima import arima

# Models
from model import category, currency, item, manufacturer, price, review, trend, forecast

NO_MANUFACTURER = 'no_manufacturer'


def start_algorithm(sc: SparkContext, config):
    items, currencies = get_parsed_items(sc)
    # for itm in items:
    data_frames_combinations = feature_dict_creator(items[0], currencies)
    final_data_frame = get_global_data_frame(data_frames_combinations)
    final_combinations = get_all_possible_combinations_from_features(final_data_frame)
    final_dictionary_data_frame = get_final_data_frames_dictionary(final_data_frame, final_combinations)
    for attr, value in final_dictionary_data_frame.items():
        arima.test_arima(attr, value)
    best_result = arima.plot_best_result()
    forecast.save_forecast(items[0].item, best_result, config)


def get_parsed_items(sc: SparkContext) -> (List[item.Item], any):

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
    return final_items, currencies


def get_all_possible_combinations_from_features(data_frame):
    column_combinations = list(data_frame.columns)
    column_combinations = [k for k in column_combinations if (k not in ['price', 'item', 'date'] and '_x' not in k and '_y' not in k)]
    all_combinations = list(chain(*map(lambda x: combinations(column_combinations, x), range(0, len(column_combinations)+1))))
    final_combinations = []
    for t in all_combinations:
        tmp = list(t)
        tmp.extend(['price', 'item', 'date'])
        final_combinations.append(tmp)
    return final_combinations


def get_final_data_frames_dictionary(original_data_frame, column_combinations):
    key_data_frame = {}
    for combination in column_combinations:
        feature_name = ",".join(combination)
        key_data_frame[feature_name] = {'data_frame': get_data_frame_from_column_names_and_original_data_frame(combination, original_data_frame), 'score': 0}
    return key_data_frame


def get_data_frame_from_column_names_and_original_data_frame(column_names, original_data_frame):
    return original_data_frame[column_names]


def feature_dict_creator(itm: item.Item, currencies: any):
    data_frames_all_features_dict = {}
    for attr, value in itm.__dict__.items():
        if attr not in ['item', 'manufacturer'] and len(value) > 0:
            data_frames_all_features_dict[attr] = get_pandas_data_frame_from_list({attr: value})
        if attr in ['item', 'manufacturer'] and value != NO_MANUFACTURER:
            data_frames_all_features_dict[attr] = get_pandas_data_frame_from_dictionary({attr: [value]})
    # All features dictionary
    for attr, value in currencies.items():
        data_frames_all_features_dict[attr] = get_pandas_data_frame_from_list({attr: value})
    return data_frames_all_features_dict


def get_global_data_frame(current_features):
    tmp_df = pd.DataFrame()
    if current_features:
        tmp_df = join_data_frames_by_optional_key(key='item', df1=current_features['item'], df2=current_features['prices'])
        for feature_key in current_features.keys():
            if feature_key != 'item' and feature_key != 'prices':
                if 'manufacturer' in current_features:
                    tmp_df = join_data_frames_by_optional_key(df1=current_features['manufacturer'], df2=tmp_df)
                    if 'trend' in current_features:
                        tmp_df = join_data_frames_by_optional_key(key=['manufacturer', 'date'], df1=current_features['trend'], df2=tmp_df)
                if 'categories' in current_features:
                    tmp_df = join_data_frames_by_optional_key(key='item', df1=current_features['categories'],
                                                              df2=tmp_df)
                if 'reviews' in current_features:
                    tmp_df = join_data_frames_by_optional_key(key=['item', 'date'], df1=current_features['reviews'],
                                                              df2=tmp_df)
                if 'euro' in current_features:
                    tmp_df = join_data_frames_by_optional_key(key='date', df1=current_features['euro'],
                                                              df2=tmp_df)
                if 'dollar' in current_features:
                    tmp_df = join_data_frames_by_optional_key(key='date', df1=current_features['dollar'],
                                                              df2=tmp_df)
    return tmp_df


def join_data_frames_by_optional_key(**kwargs):
    key = kwargs.get('key', None)
    df1 = kwargs.get('df1', None)
    df2 = kwargs.get('df2', None)
    if not key:
        df1['tmp'] = 1
        df2['tmp'] = 1
        tmp = pd.merge(df1, df2, on=key)
        del tmp['tmp']
        return tmp
    return pd.merge(df1, df2, on=key)


def get_key_name_by_object(obj: any) -> str:
    final_key_name = ''
    for key in list(obj.keys()):
        final_key_name += key + ','
        if isinstance(obj[key], pd.DataFrame) and len(obj[key]) > 2:
            final_key_name = final_key_name[:-1]
            final_key_name_as_list = final_key_name.split(',')[:-1]
            final_key_name = ','.join(final_key_name_as_list) + ','
            for inner_key in list(obj[key].keys()):
                final_key_name += inner_key + ','
    final_key_name = final_key_name[:-1]
    return final_key_name


def get_pandas_data_frame_from_list(generic_entry: any) -> pd.DataFrame:
    first_dict_key = list(generic_entry.keys())[0]
    return pd.DataFrame(data=generic_entry[first_dict_key], columns=generic_entry[first_dict_key][0]._fields)


def get_pandas_data_frame_from_dictionary(generic_dictionary: any) -> pd.DataFrame:
    return pd.DataFrame(data=generic_dictionary)
