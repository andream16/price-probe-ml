from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from model import category, currency, item, manufacturer, price, review, trend


def init_configuration(app_configuration):
    conf = SparkConf()
    conf.setMaster("local[*]")
    conf.setAppName("My application")
    conf.set("spark.driver.extraClassPath", "configuration/postgresql-42.1.4.jar")
    conf.set(" spark.executor.extraClassPath", "configuration/postgresql-42.1.4.jar")
    conf.set("spark.sql.execution.arrow.enable", "true")
    conf.set("spark.executor.memory", "1g")
    sc = SparkContext(conf=conf)
    sc_sql = SparkSession(sc)
    url = app_configuration[2]+app_configuration[0]+":"+app_configuration[1]+'/priceprobe'
    return sc_sql, url


def init_tables(sc, url):
    item.set_item_table(sc, url)
    price.set_price_table(sc, url)
    category.Category.set_category_item_table(sc, url)
    currency.set_currency_table(sc, url)
    manufacturer.set_manufacturer_table(sc, url)
    review.set_review_table(sc, url)
    trend.set_trend_table(sc, url)
