from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession


def init_configuration(app_configuration):
    conf = SparkConf()
    conf.setMaster("local")
    conf.setAppName("My application")
    conf.set('spark.jars', 'configuration/postgresql-42.1.4.jar')
    conf.set("spark.executor.memory", "1g")
    sc = SparkContext(conf=conf)
    sc_sql = SparkSession(sc)
    url = app_configuration[2]+app_configuration[0]+":"+app_configuration[1]+'/priceprobe'
    df = sc_sql.read \
        .format("jdbc") \
        .option("url", url) \
        .option("dbtable", "trend") \
        .load()
    df.printSchema()

