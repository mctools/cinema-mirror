#!/usr/bin/env python3

import sys
import os
from re import T
import numpy as np
from Cinema.PiXiu.PhononCalc.task import MeshCell, MeshQE
from Cinema.Interface.Utils import findData
import argparse
import operator
#from Cinema.Interface.parallelutil import ParallelHelper


#parameters
#temperature in kelvin
#upper limit for the Q, maxQ, float
#frequency bin size for the histogram, freSize, int
#Q bin size for the histogram, QSize, int
#stepping for the hkl, int
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--temperature', action='store', type=float,
                    dest='temp', help='temperature in kelvin', required=True)
parser.add_argument('-u', '--upper-limit-Q', action='store', type=float, default=10.0,
                    dest='maxQ', help='upper limit for the Q')
parser.add_argument('-f', '--frequency-bin-size', action='store', type=int, default=200,
                    dest='freSize', help='frequency bin size for the histogram')
parser.add_argument('-q', '--Q-bin-size', action='store', type=int, default=300,
                    dest='QSize', help='Q bin size for the histogram')
parser.add_argument('-o', '--output-file-name', action='store', default='qehist.h5',
                    dest='output', help='output file name')
parser.add_argument('-p', '--partitions', action='store', default=4,
                    dest='partitions', help='number of partitions')

args=parser.parse_args()

temp = args.temp #temperature in kelvin
maxQ = args.maxQ
freSize = args.freSize
QSize = args.QSize
output = args.output

class PhononInspect():
    def __init__(self, fn, plot=False):
        import h5py
        from Cinema.Interface.units import THz, hbar
        hf = h5py.File(fn,'r')
        self.en=hf['frequency'][()]*THz*2*np.pi*hbar
        self.w=hf['weight'][()]
        self.pmesh=hf['mesh'][()]
        self.phonnum = self.en.shape[0]
        if plot:
            import matplotlib.pyplot as plt
            for i in range(6):
                h, edges = np.histogram(self.en[:, i], bins=100, range=[0, self.en.max()], weights=self.w.reshape(-1), density=True)
                plt.plot(np.diff(edges)*0.5+edges[:-1], h)
            plt.show()
        hf.close()

    def acoustic(self):
        pass


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
            _sc.parallelize(np.random.randint(0, 100, size=(100, 3))).map(np.sum).reduce(operator.add)
            _n = _sc.defaultParallelism
        return _n
    else:
        try:
            return len(os.sched_getaffinity(0))  # only works on Linux
        except AttributeError:
            return int(os.cpu_count()) # fallback

def do_calc_msd(idx):  # idx: slice(start, end)
    calc = MeshQE('mesh.hdf5', 'out_relax.xml', temp, idx)
    return calc.calmsd()

pi = PhononInspect('mesh.hdf5')
print(f'Total phonon number: {pi.phonnum}')
num_loop = 1 # this should be the number of available cpu

USE_SPARK = False
try:
    import pyspark
    USE_SPARK = True
except ImportError:
    print('Spark not found, run in degrade mode.')

spark = None

if USE_SPARK:
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.getOrCreate()
    num_loop = get_core_count(spark)
else:
    num_loop = get_core_count()

print('Total available cores: ', num_loop)

phonEachLoop = pi.phonnum // num_loop
idx_list = [slice(i * phonEachLoop, (i + 1) * phonEachLoop) for i in range(num_loop)]

# calculate the mean squared displacement
print('Calculate MSD ...')
thermal_disp = None

if USE_SPARK:
    thermal_disp = spark.sparkContext.parallelize(idx_list, num_loop).map(do_calc_msd).reduce(operator.add)
else:
    for idx in idx_list:
        calc = MeshQE('mesh.hdf5', 'out_relax.xml', temp, idx)
        print(idx, 'calc.isoMsd()', calc.isoMsd())
        if thermal_disp is not None:
            thermal_disp +=  calc.calmsd()
        else:
            thermal_disp =  calc.calmsd()
print('calc.calmsd()', thermal_disp)  # shape: (2, 3, 3)


# calc.savePowerSqw(output,  q, en_neg, sqw, sqw_inco)
# calculate S(q, omega)
print('Calculate S(q, w) ...')
def do_calc_sqw(idx):
    calc = MeshQE('mesh.hdf5', 'out_relax.xml', temp, idx)
    calc.configHistgrame(maxQ, freSize, QSize, msd=thermal_disp)
    _q, _en_neg, _sqw, _sqw_inco = calc.getSqw()

    return [_q, _en_neg, _sqw, _sqw_inco]

def do_reduce(x, y):
    return x[0], x[1], x[2] + y[2], x[3]


q=None
en_neg=None
sqw=None
sqw_inco=None

if USE_SPARK:
    q, en_neg, sqw, sqw_inco = spark.sparkContext.parallelize(idx_list, num_loop).map(do_calc_sqw).reduce(do_reduce)
else:
    for idx in idx_list:
        # calc = MeshCell(findData('Al/mesh.hdf5'), findData('Al/cell.json'), kt)
        calc = MeshQE('mesh.hdf5', 'out_relax.xml', temp, idx)
        calc.configHistgrame(maxQ, freSize, QSize, msd=thermal_disp)
        _q, _en_neg, _sqw, _sqw_inco = calc.getSqw()
        if q is None:
            q = _q
            en_neg = _en_neg
            sqw = _sqw
            sqw_inco = _sqw_inco
        else:
            sqw += _sqw

calc = MeshQE('mesh.hdf5', 'out_relax.xml', temp, idx_list[-1])
calc.configHistgrame(maxQ, freSize, QSize, msd=thermal_disp)
calc.savePowerSqw(output,  q, en_neg, sqw, sqw_inco)

if USE_SPARK:
    spark.stop()

