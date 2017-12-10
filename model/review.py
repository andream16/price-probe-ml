def set_review_table(sc, url):

    df = sc.read.format("jdbc").options(
        url=url,
        driver='org.postgresql.Driver',
        dbtable='review',
        user='postgres'
        # password='your_password')
    ).load()
    df.createOrReplaceTempView("review")