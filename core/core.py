# 3rd party
from pyspark import SparkContext

# Models
from model import category, currency, item, manufacturer, price, review, trend


def start_algorithm(sc: SparkContext):

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
                if current_manufacturer != 'no_manufacturer':
                    current_trend = trend.get_trend_by_manufacturer(sc, current_manufacturer)
            current_prices = price.Price.get_prices_by_item(sc, current_item_id)
            current_reviews = []
            if i.has_reviews:
                current_reviews = review.Review.get_reviews_by_item(sc, current_item_id)
            current_categories = category.get_categories_by_item(current_item_id, sc)
            final_items.append(item.Item(current_item_id, current_categories, current_manufacturer, current_prices,
                                         current_reviews, current_trend))
