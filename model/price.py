from collections import OrderedDict


def set_price_table(sc, url):

    df = sc.read.format("jdbc").options(
        url=url,
        driver='org.postgresql.Driver',
        dbtable='price',
        user='postgres'
        # password='your_password')
    ).load()
    df.createOrReplaceTempView("price")


# class Price(tuple):
#     __slots__ = ()
#
#     _fields = ('item', 'date', 'price', 'flag')
#
#     def __new__(_cls, item, date, price, flag):
#         'Create new instance of Price'
#         return _tuple.__new__(_cls, (item, date, price, flag))
#
#     @classmethod
#     def _make(cls, iterable, new=tuple.__new__, len=len):
#         'Make a new Price object from a sequence or iterable'
#         result = new(cls, iterable)
#         if len(result) != 4:
#             raise TypeError('Expected 4 arguments, got %d' % len(result))
#         return result
#
#     def _asdict(self):
#         'Return a new OrderedDict which maps field names to their values'
#         return OrderedDict(zip(self._fields, self))
#
#     def _replace(_self, **kwds):
#         'Return a new Price object replacing specified fields with new values'
#         result = _self._make(map(kwds.pop, ('item', 'date', 'price', 'flag'), _self))
#         if kwds:
#             raise ValueError('Got unexpected field names: %r' % kwds.keys())
#         return result
