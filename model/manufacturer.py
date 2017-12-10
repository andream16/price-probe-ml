def set_manufacturer_table(sc, url):

    df = sc.read.format("jdbc").options(
        url=url,
        driver='org.postgresql.Driver',
        dbtable='manufacturer',
        user='postgres'
        # password='your_password')
    ).load()
    df.createOrReplaceTempView("manufacturer")