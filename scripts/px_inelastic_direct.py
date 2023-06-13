#!/usr/bin/env python3

from re import T
import numpy as np
from Cinema.PiXiu.PhononCalc.task import MeshCell, MeshQE
from Cinema.Interface.Utils import findData
import argparse

from Cinema.Interface.parallelutil import ParallelHelper
from pyspark import SparkContext

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

# ph = ParallelHelper()
# ph.partitions = args.partitions
# ph.sparkContext = SparkContext(master=f'local[{ph.partitions}]')


temp = args.temp #temperature in kelvin
maxQ = args.maxQ
freSize = args.freSize
QSize = args.QSize
output = args.output

class PhononInspect:
    def __init__(self, fn, plot=False):
        import h5py
        import matplotlib.pyplot as plt
        from Cinema.Interface.units import THz, hbar
        hf = h5py.File(fn,'r')
        self.en=hf['frequency'][()]*THz*2*np.pi*hbar
        self.w=hf['weight'][()]
        self.phonnum = self.en.shape[0]
        if plot:
            for i in range(6):
                h, edges = np.histogram(self.en[:, i], bins=100, range=[0, self.en.max()], weights=self.w.reshape(-1), density=True)
                plt.plot(np.diff(edges)*0.5+edges[:-1], h)
            plt.show()
        hf.close()

    def acoustic(self):
        pass


pi = PhononInspect('mesh.hdf5')
# num_loop = 100
# phonEachLoop = pi.phonnum//num_loop

print(pi.phonnum)

calc = MeshQE('mesh.hdf5', 'out_relax.xml', temp, slice(0,-1))
print('calc.isoMsd()', calc.isoMsd())

thermal_disp =  calc.calmsd()
print('calc.calmsd()', thermal_disp)
print('ratio', calc.isoMsd()/thermal_disp.sum())






# q=None
# en_neg=None
# sqw=None
# sqw_inco=None
# for i in range(num_loop):
#     # calc = MeshCell(findData('Al/mesh.hdf5'), findData('Al/cell.json'), kt)
#     calc = MeshQE('mesh.hdf5', 'out_relax.xml', temp, slice(phonEachLoop*i,phonEachLoop*(i+1)))
#     calc.configHistgrame(maxQ, freSize, QSize)
#     _q, _en_neg, _sqw, _sqw_inco = calc.getSqw()
#     if q is None:
#         q = _q
#         en_neg = _en_neg
#         sqw = _sqw
#         sqw_inco = _sqw_inco
#     else:
#         sqw += _sqw

# calc.savePowerSqw(output,  q, en_neg, sqw, sqw_inco)


