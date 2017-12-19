from collections import namedtuple

Category = namedtuple("Category", "category item")


def set_category_table(sc, url):
    df = sc.read.format("jdbc").options(
        url=url,
        driver='org.postgresql.Driver',
        dbtable='category',
        user='postgres'
    ).load()
    df.createOrReplaceTempView("category")