def set_item_table(sc, url):

    df = sc.read.format("jdbc").options(
        url=url,
        driver='org.postgresql.Driver',
        dbtable='item',
        user='postgres'
    ).load()
    df.createOrReplaceTempView("item")

