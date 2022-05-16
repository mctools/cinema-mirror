#!/usr/bin/env python3

from Cinema.PiXiu import getAtomMassBC, Pseudo, QEType
from Cinema.PiXiu.IO import JsonCell, XmlCell
import numpy as np
import os, sys, glob, re
from loguru import logger

logger.add(sys.stdout, colorize=True, format="<green>{time}</green> <level>{message}</level>", backtrace=True, level="INFO")
logger.add("file.log", rotation="12:00")

def lastGoodNumber(n):
    return int(2**np.floor(np.log2(n)))

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', action='store', type=str, default='mp-13_Fe.json',
                    dest='input', help='input json file')
parser.add_argument('-n', '--numcpu', action='store', type=int, default=lastGoodNumber(os.cpu_count()//2),
                    dest='numcpu', help='number of CPU')
parser.add_argument('-m', '--magcut', action='store', type=float, default=1.0,
                    dest='magcut', help='total magnetization')
parser.add_argument('-r', '--rundft', action='store_true', dest='rundft', help='run DFT')

args = parser.parse_args()
inputfile=args.input
cores=args.numcpu
rundft=args.rundft

logger.info(f'px.py input {inputfile}, CPU {cores}, {rundft} run DFT')
ps = Pseudo()

pcell = True
unitcell_sim="unitcell.in"
unitcellrelex_sim="unitcell_rl.in"

c = JsonCell(inputfile)
dim = c.estSupercellDim()
kpt_relax = c.estRelaxKpoint()
kpt = c.estSupercellKpoint()
cell = c.getCell()
magn = c.getTotalMagnetic()
sgnum = c.getSpacegourpNum()

logger.info(f'total magnetization: {magn}, spacegroup: {sgnum}')

if magn is not None:
    if magn>args.magcut:
        logger.info(f'total magnetization: {magn}, cutoff: {magcut.magcut}, skipping this material')
        sys.exit()
    else:
        logger.warning('total_magnetization field is not sepecified in the input json file')

logger.info(f'original cell {cell}, kpoint for relax {kpt_relax}')

spacegroup, lattice , positions, elements = ps.qems(cell, unitcellrelex_sim, dim, kpt, QEType.Relax, usePrimitiveCell=pcell )
qxsg = int(re.findall(r"\(\s*\+?(-?\d+)\s*\)", spacegroup)[0])
logger.info(f'space group {spacegroup}')

if sgnum != qxsg:
    logger.critical(f'space group in the origianl input is inconsistent with that idealised by PX')
    sys.exit()

if not os.path.isfile('out.xml'):
    if os.system(f'mpirun -np {cores} pw.x -nk {cores//4} -inp {unitcellrelex_sim} | tee {unitcellrelex_sim[:-3]}.out' ):
        raise IOError("Relax pw.x fail")


relexed_cell = XmlCell('out.xml')
dim = relexed_cell.estSupercellDim()
kpt = relexed_cell.estSupercellKpoint()
cell = relexed_cell.getCell()
mesh =  relexed_cell.estMesh()

logger.info(f'cell after relax {cell}')

spacegroup, lattice , positions, elements = ps.qems(cell, unitcell_sim, dim, kpt, QEType.Scf, usePrimitiveCell=pcell )
qxsgsg = int(re.findall(r"\(\s*\+?(-?\d+)\s*\)", spacegroup)[0])

if sgnum != qxsgsg:
    logger.critical(f'space group in the origianl input{sgnum} is inconsistent with that of idealised supercell {qxsgsg} by PX')
    sys.exit()

logger.info(f'supercell info {cell}, mesh {mesh}, kpt {kpt}, dim {dim}')

if rundft:
    fs=glob.glob('supercell-*.in')
    fs=sorted(fs)
    for f in fs:
        if os.system(f'mpirun -np {cores} pw.x -inp {f} | tee {f[0:-3]}.out' ):
            raise IOError("SCF pw.x fail")

    if os.system('phonopy --qe -f supercell-*.out' ):
        raise IOError("force fail")

    #density of states
    if os.system(f'phonopy -v --qe -c unitcell.in --dim {dim[0]} {dim[1]} {dim[2]} --pdos AUTO --mesh {mesh[0]} {mesh[1]} {mesh[2]}   --nowritemesh '): # -p
        raise IOError("dos fail")

    #mesh
    if os.system(f'phonopy --qe -c unitcell.in --dim {dim[0]} {dim[1]} {dim[2]}  --mesh {mesh[0]} {mesh[1]} {mesh[2]} --hdf5-compression gzip --hdf5  --eigvecs --nomeshsym'):
        raise IOError("mesh fail")

logger.info(f'Calculation completed!')
