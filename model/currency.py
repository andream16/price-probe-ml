def set_currency_table(sc, url):

    df = sc.read.format("jdbc").options(
        url=url,
        driver='org.postgresql.Driver',
        dbtable='currency',
        user='postgres'
        # password='your_password')
    ).load()
    df.createOrReplaceTempView("currency")