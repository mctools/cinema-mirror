#!/usr/bin/env python3

import sys
import os
from re import T
import numpy as np
from Cinema.PiXiu.PhononCalc.task import MeshCell, MeshQE
from Cinema.Interface.Utils import findData
import argparse
import operator
from Cinema.Interface import cimspark

#from Cinema.Interface.parallelutil import ParallelHelper


def gen_parser():
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
    parser.add_argument('-p', '--partitions', action='store', type=int, default=0,
                        dest='partitions', help='number of partitions. 0: automatic')
    return parser
    
args=gen_parser().parse_args()
temp = args.temp #temperature in kelvin
maxQ = args.maxQ
freSize = args.freSize
QSize = args.QSize
output = args.output
partitions = args.partitions

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


pi = PhononInspect('mesh.hdf5')
print(f'Total phonon number: {pi.phonnum}')
num_loop = 1 # this should be the number of available cpu


phonEachLoop = pi.phonnum // num_loop
if partitions <= 2:
    partitions = num_loop
idx_list = [slice(i * phonEachLoop, (i + 1) * phonEachLoop) for i in range(num_loop)]

# calculate the mean squared displacement
print('Calculate MSD ...')

@cimspark.parallelize(idx_list, operator.add, mode='auto')
def getMSD(idx):
    return MeshQE('mesh.hdf5', 'out_relax.xml', temp, idx).calmsd()

thermal_disp = getMSD(300)
print('calc.calmsd()', thermal_disp)  # shape: (2, 3, 3)


# calculate S(q, omega)
print('Calculate S(q, w) ...')

@cimspark.parallelize(idx_list, lambda x, y: [ x[0], x[1], x[2] + y[2], x[3] ], mode='auto')
def do_calc_sqw(idx):
    calc = MeshQE('mesh.hdf5', 'out_relax.xml', temp, idx)
    calc.configHistgrame(maxQ, freSize, QSize, msd=thermal_disp)
    _q, _en_neg, _sqw, _sqw_inco = calc.getSqw()
    return _q, _en_neg, _sqw, _sqw_inco


q, en_neg, sqw, sqw_inco = do_calc_sqw()

calc = MeshQE('mesh.hdf5', 'out_relax.xml', temp, idx_list[-1])
calc.configHistgrame(maxQ, freSize, QSize, msd=thermal_disp)
calc.savePowerSqw(output,  q, en_neg, sqw, sqw_inco)

