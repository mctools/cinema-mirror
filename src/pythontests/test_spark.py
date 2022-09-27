#!/usr/bin/env python3

from pyspark import SparkContext
import numpy as np

from Cinema.PiXiu.PhononCalc.calcBase import PowderHKLIter

import itertools
from pyspark.serializers import MarshalSerializer

def dummy(x):
    print(f'function received parameter {x}')
    return x


# stand-alone
it = PowderHKLIter(np.eye(3), 2)
res1 = list(map(dummy, it))

# spark
it = PowderHKLIter(np.eye(3), 2)
sc=SparkContext(master="local[4]")
A=sc.parallelize(it).map(dummy)
print(A.glom().collect())
res2 = A.collect()

import hashlib, pickle
m1=hashlib.md5(pickle.dumps(res1))
m2=hashlib.md5(pickle.dumps(res2))
print(m1.hexdigest(), m2.hexdigest()) # the order is not the same

np.testing.assert_allclose(17, len(res1))
