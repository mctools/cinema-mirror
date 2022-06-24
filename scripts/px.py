#!/usr/bin/env python3

from Cinema.PiXiu import getAtomMassBC, Pseudo, QEType
from Cinema.PiXiu.IO import JsonCell, XmlCell
import numpy as np
import os, sys, glob, re
from loguru import logger
import seekpath

def getPath(path):
    cor = path['point_coords']
    pcor=[]
    label=[]
    for p in path['path']:
        begin = p[0]
        end = p[1]
        pcor.append(cor[begin])
        pcor.append(cor[end])
        label.append(begin)
        label.append(end)
    pcor=np.array(pcor).flatten()
    for i in range(len(label)):
        if label[i]=='GAMMA':
            label[i] = '$\Gamma$'
        # elif '_' in label[i]:
        #     label[i] = '$'+str(label[i]) + '$'
    return pcor, label


logger.add(sys.stdout, colorize=True, format="<green>{time}</green> <level>{message}</level>", backtrace=True, level="INFO")
logger.add("pixiu.log", rotation="12:00")

def lastGoodNumber(n):
    return int(2**np.floor(np.log2(n)))

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', action='store', type=str, default='mp-13_Fe.json',
                    dest='input', help='input json file')
parser.add_argument('-n', '--numcpu', action='store', type=int, default=lastGoodNumber(os.cpu_count()//2),
                    dest='numcpu', help='number of CPU')
parser.add_argument('-r', '--rundft', action='store_true', dest='rundft', help='run DFT')
parser.add_argument('-p', '--plotpdf', action='store_true', dest='plotpdf', help='generate pdf files')

args = parser.parse_args()
inputfile=args.input
cores=args.numcpu
rundft=args.rundft

plotflag=''
if args.plotpdf:
    plotflag='-p'

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
    logger.info(f'mp total magnetization: {magn}')
else:
    logger.info(f'mp info contains no total magnetization')

logger.info(f'original cell {cell}, kpoint for relax {kpt_relax}')

spacegroup, lattice , positions, elements, numbers = ps.qems(cell, unitcellrelex_sim, dim, kpt, QEType.Relax, usePrimitiveCell=pcell)
qxsg = int(re.findall(r"\(\s*\+?(-?\d+)\s*\)", spacegroup)[0])
logger.info(f'space group after standardize_cell before relaxing {spacegroup}')

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

logger.info(f'cell after relax {cell}, total magnetisation {relexed_cell.totmagn}')

spacegroup, lattice , positions, elements, numbers = ps.qems(cell, unitcell_sim, dim, kpt, QEType.Scf, usePrimitiveCell=pcell )
qxsgsg = int(re.findall(r"\(\s*\+?(-?\d+)\s*\)", spacegroup)[0])


path = seekpath.get_path((lattice , positions, numbers), with_time_reversal=1)
logger.info(f'path {path}')
logger.info(f'space group before SCF calculation {qxsgsg}')

if sgnum != qxsgsg:
    logger.critical(f'space group in the origianl input {sgnum} is inconsistent with that of idealised supercell {qxsgsg} by PX')
    sys.exit()

logger.info(f'supercell info {cell}, mesh {mesh}, kpt {kpt}, dim {dim}')

if rundft and not os.path.isfile('FORCE_SETS'):
    fs=glob.glob('supercell-*.in')
    fs=sorted(fs)
    for f in fs:
        if os.system(f'mpirun -np {cores} pw.x -inp {f} | tee {f[0:-3]}.out' ):
            logger.info(f'SCF pw.x fail')
            raise IOError("SCF pw.x fail")

    if os.system('phonopy --qe -f supercell-*.out' ):
        logger.info(f'force fail')
        raise IOError("force fail")

#band
pcor, label = getPath(path)
logger.info(f'band {" ".join(map(str,pcor))}; {" ".join(map(str,label))}')
if os.system(f'phonopy --dim "{dim[0]} {dim[1]} {dim[2]}" --band="{" ".join(map(str,pcor))}" --band-labels="{" ".join(map(str,label))}" --hdf5 --eigvecs {plotflag} -s'):
    logger.info(f'band fail')
    raise IOError("band fail")

#density of states
if os.system(f'phonopy -v --qe -c unitcell.in --dim {dim[0]} {dim[1]} {dim[2]} --pdos AUTO --mesh {mesh[0]} {mesh[1]} {mesh[2]}  --nowritemesh {plotflag} -s'):
    logger.info(f'dos fail')
    raise IOError("dos fail")

#mesh
if os.system(f'phonopy --qe -c unitcell.in --dim {dim[0]} {dim[1]} {dim[2]}  --mesh {mesh[0]} {mesh[1]} {mesh[2]} --hdf5-compression gzip --hdf5  --eigvecs --nomeshsym'):
    logger.info(f'mesh fail')
    raise IOError("mesh fail")

logger.info(f'Calculation completed!')
