import functools

def singleton(cls):
    @functools.wraps(cls)
    def inner(*args, **kwargs):
        if not inner.instance:
            inner.instance = cls(*args, **kwargs)
        return inner.instance
    inner.instance=None
    return inner

@singleton
class SparkHelper():
    sparkContext = None
    partitions = None

    def __init__(self):
        pass

    def available(self):
        if self.sparkContext and self.partitions:
            return True
        else:
            return False

    def mapReduce(self, func, iterabe1):
         if self.available():
             res = SparkHelper().sparkContext.parallelize(iterabe1, SparkHelper.partitions).map(func).reduce()
         else:
             res = list(map(func, iterabe1))
         return res
