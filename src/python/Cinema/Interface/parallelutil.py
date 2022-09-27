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
class ParallelHelper():
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
             res = self.sparkContext.parallelize(iterabe1, self.partitions).map(func).collect()
         else:
             res = list(map(func, iterabe1))
         return res
