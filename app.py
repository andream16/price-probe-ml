from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from configuration import configuration
from spark import spark
app_configuration = configuration.get_configuration('local')
spark.init_configuration(app_configuration)
