def set_trend_table(sc, url):

    df = sc.read.format("jdbc").options(
        url=url,
        driver='org.postgresql.Driver',
        dbtable='trend',
        user='postgres'
        # password='your_password')
    ).load()
    df.createOrReplaceTempView("trend")