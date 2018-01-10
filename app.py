# Spark
from spark import spark

# Configuration
from configuration import configuration

# Core
from core import core


app_configuration = configuration.get_configuration('remote')
spark = spark.Spark()
spark_context = spark.init_configuration(app_configuration)

# Core Startup
core.start_algorithm(spark_context, app_configuration)




