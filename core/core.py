from typing import List
from itertools import combinations, chain
# 3rd party
from pyspark import SparkContext
import pandas as pd

# ARIMA
from algorithm.arima import arima

# Models
from model import item, manufacturer, price, review, trend, forecast

NO_MANUFACTURER = 'no_manufacturer'


def start_algorithm(sc: SparkContext, config):
    #fout = "results2.txt"
    #fo = open(fout, "w")
    items = get_parsed_items(sc)
    for itm in items:
        results = {}
        data_frames_combinations = feature_dict_creator(itm)
        print(itm.item)
        final_data_frame = get_global_data_frame(data_frames_combinations)
        final_combinations = get_all_possible_combinations_from_features(final_data_frame)
        final_dictionary_data_frame = get_final_data_frames_dictionary(final_data_frame, final_combinations)
        arima_dict = arima.find_arima_parameters_by_dataframe(final_data_frame)
        for attr, value in final_dictionary_data_frame.items():
           arima_dict['data_frame'] = value['data_frame']
           results[attr] = arima.test_arima(attr, arima_dict)
           #lstm.test_lstm(arima_dict)

        best_results = arima.plot_best_result(results)
        for result in best_results:
            forecast.save_forecast(itm.item, result, config)
        #fo.write(str(itm.item)+' - '+str(best_result['score'])+'\n')
        #fo.write(str(best_result['score'])+'\n')
    #fo.close()


def get_parsed_items(sc: SparkContext) -> (List[item.Item], any):
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
            final_items.append(item.Item(current_item_id, current_manufacturer, current_prices,
                                         current_reviews, current_trend))
    return final_items


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


def feature_dict_creator(itm: item.Item):
    data_frames_all_features_dict = {}
    for attr, value in itm.__dict__.items():
        if attr not in ['item', 'manufacturer'] and value and len(value) > 0:
            data_frames_all_features_dict[attr] = get_pandas_data_frame_from_list({attr: value})
        if attr in ['item', 'manufacturer'] and value != NO_MANUFACTURER:
            data_frames_all_features_dict[attr] = get_pandas_data_frame_from_dictionary({attr: [value]})
    # All features dictionary
    return data_frames_all_features_dict


def get_global_data_frame(current_features):
    tmp_df = pd.DataFrame
    flag = True
    manufacturer_flag = False
    if current_features:
        tmp_df = join_data_frames_by_optional_key(key='item', df1=current_features['item'], df2=current_features['prices'])
        for feature_key in current_features.keys():
            if feature_key != 'item' and feature_key != 'prices' and feature_key != 'trend':
                if 'manufacturer' in current_features:
                    manufacturer_flag = True
                    tmp_df = join_data_frames_by_optional_key(df1=current_features['manufacturer'], df2=tmp_df)
                    if 'trend' in current_features and flag:
                        flag = False
                        tmp_df = join_data_frames_by_optional_key(key=['manufacturer', 'date'], df1=current_features['trend'],
                                                                  df2=tmp_df)
                if 'reviews' in current_features:
                    temp_reviews_df = join_data_frames_by_optional_key(key=['item', 'date'], df1=current_features['reviews'],
                                                              df2=tmp_df)
                    if temp_reviews_df.size > 4:
                        tmp_df = temp_reviews_df.rename(columns={'sentiment_x' : 'sentiment', 'stars_x' : 'stars'})
        if manufacturer_flag:
            tmp_df = tmp_df.drop(['manufacturer'], axis=1)
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






