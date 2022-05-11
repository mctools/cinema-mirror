#!/usr/bin/env python3

from Cinema.PiXiu import getAtomMassBC, Pseudo, QEType
from Cinema.PiXiu.IO import JsonCell, XmlCell
import numpy as np
import os, sys, glob

def lastGoodNumber(n):
    return int(2**np.floor(np.log2(n)))

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', action='store', type=str, default='mp-13_Fe.json',
                    dest='input', help='input json file')
parser.add_argument('-n', '--numcpu', action='store', type=int, default=lastGoodNumber(os.cpu_count()//2),
                    dest='numcpu', help='number of CPU')

args = parser.parse_args()
inputfile=args.input
cores=args.numcpu

ps = Pseudo()

pcell = True
unitcell_sim="unitcell.in"

c = JsonCell(inputfile)
dim = c.estSupercellDim()
kpt_relax = c.estRelaxKpoint()
kpt = c.estSupercellKpoint()
cell = c.getCell()

ps.qems(cell, unitcell_sim, dim, kpt, QEType.Relax, usePrimitiveCell=pcell )


if os.system(f'mpirun -np {cores} pw.x -nk {cores//4} -inp {unitcell_sim}' ):
    raise IOError("Relax pw.x fail")


relexed_cell = XmlCell('out.xml')
dim = relexed_cell.estSupercellDim()
kpt = relexed_cell.estSupercellKpoint()
cell = relexed_cell.getCell()
mesh =  relexed_cell.estMesh()

ps.qems(cell, unitcell_sim, dim, kpt, QEType.Scf, usePrimitiveCell=pcell )

print('cell', cell)
print('mesh', mesh)
print('kpt', kpt)
print('dim', dim)

fs=glob.glob('supercell-*.in')
fs=sorted(fs)
for f in fs:
    if os.system(f'mpirun -np {cores} pw.x -inp {f} | tee {f[0:-3]}.out' ):
        raise IOError("SCF pw.x fail")

if os.system('phonopy --qe -f supercell-*.out' ):
    raise IOError("force fail")

#density of states
if os.system(f'phonopy -v --qe -c unitcell.in --dim {dim[0]} {dim[1]} {dim[2]} --pdos AUTO --mesh {mesh[0]} {mesh[1]} {mesh[2]}   --nowritemesh -p '):
    raise IOError("dos fail")

# #mesh
# if os.system(f'phonopy --qe -c unitcell.in --dim {dim[0]} {dim[1]} {dim[2]}  --mesh {mesh[0]} {mesh[1]} {mesh[2]} --hdf5-compression gzip --hdf5  --eigvecs --nomeshsym'):
#     raise IOError("mesh fail")
