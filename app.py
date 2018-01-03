# 3rd party
from spark import spark

# Configuration
from configuration import configuration

# Models
from model import currency, category

app_configuration = configuration.get_configuration('remote')
sc, url = spark.init_configuration(app_configuration)
spark.init_tables(sc, url)

category = category.Category("", [])
category.get_categories_by_item("B0019X20R8", sc)
currency = currency.Currency("", [])
currency.get_currencies(sc)
