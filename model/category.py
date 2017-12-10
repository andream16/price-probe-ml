def set_category_table(sc, url):

    df = sc.read.format("jdbc").options(
        url=url,
        driver='org.postgresql.Driver',
        dbtable='category',
        user='postgres'
        # password='your_password')
    ).load()
    df.createOrReplaceTempView("category")