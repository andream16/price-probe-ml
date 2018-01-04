# 3rd party
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession

TABLES = [
    'category_item',
    'currency',
    'item',
    'manufacturer',
    'price',
    'review',
    'trend',
]


class Spark:

    def init_configuration(self, configuration):
        conf = SparkConf()
        conf.setMaster("local[*]")
        conf.setAppName("My application")
        conf.set("spark.driver.extraClassPath", "configuration/postgresql-42.1.4.jar")
        conf.set(" spark.executor.extraClassPath", "configuration/postgresql-42.1.4.jar")
        conf.set("spark.sql.execution.arrow.enable", "true")
        conf.set("spark.executor.memory", "1g")
        sc = SparkSession(SparkContext(conf=conf))
        url = configuration[2]+configuration[0]+":"+configuration[1]+'/priceprobe'
        self.init_tables(sc, url)
        return sc

    def init_tables(self, sc, url):
        for table in TABLES:
            self.set_table_by_name(sc, url, table)

    @staticmethod
    def set_table_by_name(sc: SparkContext, url: str, table_name: str):
        df = sc.read.format("jdbc").options(
            url=url,
            driver='org.postgresql.Driver',
            dbtable=table_name,
            user='postgres'
        ).load()
        df.createOrReplaceTempView(table_name)
