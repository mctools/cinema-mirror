#!/usr/bin/env python3

from Cinema.PiXiu import getAtomMassBC, Pseudo, QEType
from Cinema.PiXiu.io import MPCell, QeXmlCell
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

def get_material_from_db(mp_id, db_url, db='cinema', coll='mp'):
    import pymongo

    client = pymongo.MongoClient(db_url)
    mp = client[db][coll]

    return mp.find_one({'mp_id': mp_id})

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
parser.add_argument('-t', '--thermal', action='store_true', dest='thermal', help='calculate thermal displacement')
parser.add_argument('-p', '--plotpdf', action='store_true', dest='plotpdf', help='generate pdf files')
parser.add_argument('-m', '--mp-id', action='store', type=str, default=None,
                    dest='mpid', help='the Materials Project ID. This option will take precedence than --input. mp-149 for example.')
parser.add_argument('-c', '--mongo-url', action='store', type=str, default=None,
        dest='mongo_url', help='the MongoDB connection string. mongodb://user:pass@localhost:27017/ for example.')
parser.add_argument('-s', '--use-spark', action='store_true', dest='use_spark', help='run in spark')
parser.add_argument('-v', '--vdW', action='store_true', dest='vdW', help='consider Van Der Waals forces in DFT')
parser.add_argument('-d', '--phonmeshdensity', action='store', dest='phonmeshdensity', type=float, default=100., help='Est phonon mesh density per atom per 1/Aa')

args = parser.parse_args()
inputfile=args.input
cores=args.numcpu
rundft=args.rundft
mpid = args.mpid
vdW = args.vdW
thermal = args.thermal
phonmeshdensity = args.phonmeshdensity

use_spark = args.use_spark
mongo_url = args.mongo_url

if mpid is not None and len(mpid.strip()) == 0:
    logger.error('input mpid must not be empty.')
    sys.exit(0)

if mpid is not None:
    if mongo_url is None or len(mongo_url.strip()) == 0:
        logger.error('mongo url must be provided with mpid')
        sys.exit(0)
    else:
        mp_doc = get_material_from_db(mpid, mongo_url)
        if mp_doc is None:
            logger.error(f'{mpid} not found in the database.')
            sys.exit(0)
        inputfile = mp_doc

if use_spark:
    logger.info('Run as Spark application')
    import pyspark
    from pyspark.sql import SparkSession

    if mpid is None:
        pf = os.path.basename(inputfile)
    else:
        pf = mpid
    spark = SparkSession.builder.appName(f'DFT_Job_{pf}').getOrCreate()
    #mode = spark.conf.get('spark.submit.deployMode')

plotflag=''
if args.plotpdf:
    plotflag='-p'

qe_nk=''
if cores>4:
    qe_nk = f'-nk {cores//4}'

if mpid is None:
    logger.info(f'px.py input {inputfile}, CPU {cores}, {rundft} run DFT')
else:
    logger.info(f'px.py input {mpid}, CPU {cores}, {rundft} run DFT')

ps = Pseudo()

pcell = True
scf_qeinput="unitcell.in"
relex_qeinput="unitcell_rl.in"

c = MPCell(inputfile)
logger.info(f'px.py read json cell info, abc: {c.abc}, lattice: {c.lattice}')
cell = c.getCell()
magn = c.getTotalMagnetic()
sgnum = c.getSpacegourpNum()

logger.info(f'total magnetization: {magn}, spacegroup: {sgnum}')

if magn is not None:
    logger.info(f'mp total magnetization: {magn}')
else:
    logger.info(f'mp info contains no total magnetization')

logger.info(f'original cell {cell}')

spacegroup, lattice , positions, elements, numbers = ps.qems(cell, relex_qeinput, QEType.Relax, usePrimitiveCell=pcell, vdW=vdW)
qxsg = int(re.findall(r"\(\s*\+?(-?\d+)\s*\)", spacegroup)[0])
logger.info(f'space group after standardize_cell before relaxing {spacegroup}')

if sgnum != qxsg:
    logger.critical(f'space group in the origianl input is inconsistent with that idealised by PX')
    sys.exit()

if not os.path.isfile('out_relax.xml'):
    if os.system(f'mpirun -np {cores} pw.x {qe_nk} -inp {relex_qeinput} | tee {relex_qeinput[:-3]}.out' ):
        raise IOError("Relax pw.x fail")
    os.rename('out.xml', 'out_relax.xml')

#out_relax.xml containes the calculated results for the relaxed cell

relexed_cell = QeXmlCell('out_relax.xml')
dim = relexed_cell.estSupercellDim()
kpt = relexed_cell.estSupercellKpoint()
cell = relexed_cell.getCell()
mesh =  relexed_cell.estPhonMesh(phonmeshdensity)

logger.info(f'cell after relax {cell}, total magnetisation {relexed_cell.totmagn}')

spacegroup, lattice , positions, elements, numbers = ps.qems(cell, scf_qeinput, QEType.Scf, usePrimitiveCell=pcell, vdW=vdW )
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
if os.system(f'phonopy --fc-symmetry  --dim "{dim[0]} {dim[1]} {dim[2]}" --band="{" ".join(map(str,pcor))}" --band-labels="{" ".join(map(str,label))}" --hdf5 --eigvecs {plotflag} -s --writefc'):
    logger.info(f'band fail')
    raise IOError("band fail")

#tdm
if thermal:
    if os.system(f'phonopy --fc-symmetry --qe -c unitcell.in --dim {dim[0]} {dim[1]} {dim[2]}  --tdm --mesh {mesh[0]} {mesh[1]} {mesh[2]}'):
        logger.info(f'dos and mesh fail')
        raise IOError("dos and mesh fail")

#mesh
if os.system(f'phonopy --fc-symmetry --qe -c unitcell.in --dim {dim[0]} {dim[1]} {dim[2]}   --mesh {mesh[0]} {mesh[1]} {mesh[2]} --hdf5-compression gzip --hdf5  --eigvecs --nomeshsym --pdos AUTO  {plotflag} -s'):
    logger.info(f'dos and mesh fail')
    raise IOError("dos and mesh fail")

logger.info(f'Calculation completed!')

if use_spark:
    logger.info('Stop Spark.')
    spark.stop()
