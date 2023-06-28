
import functools
import os
import numpy as np

try:
    import pyspark as spark
    USE_SPARK = True
except ImportError:
    print('Spark not found, run in degrade mode.')


def get_core_count(spark_session=None):
    if spark_session is not None:
        _sc = spark_session.sparkContext
        _app_id = _sc.getConf().get('spark.app.id')

        if _app_id.startswith('local-'):
            return _sc.defaultParallelism   # local spark
    
        _n = _sc.defaultParallelism
        try:
            _n = int(spark.sparkContext.getConf().get('spark.cores.max')) # for spark cluster with core count specified
            return _n
        except:
            pass
        if _n == _sc.defaultParallelism:  # for spark cluster without core count specified
            import operator
            _sc.parallelize(np.random.randint(0, 100, size=(100, 3))).map(np.sum).reduce(operator.add)
            _n = _sc.defaultParallelism
        return _n
    else:
        try:
            return len(os.sched_getaffinity(0))  # only works on Linux
        except AttributeError:
            return int(os.cpu_count()) # fallback


def parallelize(iteratable, oper, mode='auto'):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            print(*args, **kwargs)
            if USE_SPARK == False or mode is None:
                return functools.reduce(oper, map(func, iteratable))

            elif USE_SPARK and mode== 'auto':
                from pyspark.sql import SparkSession
                spark = SparkSession.builder.getOrCreate()
                p = get_core_count(spark)
                res = spark.sparkContext.parallelize(iteratable, p).map(func).reduce(oper)
                spark.stop()
                return res
        return wrapper
    return decorator

        

