#!/usr/bin/env python3

from Cinema.PiXiu import getAtomMassBC, Pseudo, QEType
from Cinema.PiXiu.IO import JsonCell, XmlCell
import numpy as np
import os, sys, os
import json, copy

cores=os.cpu_count()//2
ps = Pseudo()

pcell = True
unitcell_sim="unitcell.in"

c = JsonCell('mp-1271562_Fe.json')
dim = c.estSupercellDim()
kpt_relax = c.estRelaxKpoint()
kpt = c.estSupercellKpoint()
cell = c.getCell()

ps.qems(cell, unitcell_sim, dim, kpt, QEType.Relax, usePrimitiveCell=pcell )


if os.system(f'mpirun -np {cores} pw.x -nk {cores//4} -inp {unitcell_sim}' ):
    raise IOError("Relax pw.x fail")


relexed_cell = XmlCell('out.xml')
dim = relexed_cell.estSupercellDim()
kpt_relax = relexed_cell.estRelaxKpoint()
kpt = relexed_cell.estSupercellKpoint()
cell = relexed_cell.getCell()
mesh =  relexed_cell.estMesh()

ps.qems(cell, unitcell_sim, dim, kpt, QEType.Scf, usePrimitiveCell=pcell )

print('mesh', mesh)


if os.system(f'mpirun -np {cores} pw.x -nk {cores//4} -inp {f} | tee {f[0:-3]}.out' ):
    raise IOError("SCF pw.x fail")

if os.system('phonopy -f supercell-*.out' ):
    raise IOError("force fail")

#density of states
if os.system(f'phonopy -v --qe -c unit_cell.in --dim {dim[0]} {dim[1]} {dim[2]} --pdos AUTO --mesh {mesh[0]} {mesh[1]} {mesh[2]}   --nowritemesh -p '):
    raise IOError("dos fail")

#mesh
if os.system(f'phonopy --qe -c unit_cell.in --dim {dim[0]} {dim[1]} {dim[2]}  --mesh {mesh[0]} {mesh[1]} {mesh[2]} --hdf5-compression gzip --hdf5  --eigvecs --nomeshsym'):
    raise IOError("mesh fail")
