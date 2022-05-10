#!/usr/bin/env python3

from Cinema.PiXiu import getAtomMassBC, Pseudo, QEType
from Cinema.PiXiu.IO import Cell
import numpy as np
import os, sys, os
import json, copy

pcell = True
unitcell_sim="unitcell.in"

c = Cell('mp-1271562_Fe.json')
dim = c.estSupercellDim()
kpt_relax = c.estRelaxKpoint()
kpt = c.estSupercellKpoint()
cell = c.getCell()

ps = Pseudo()
ps.qems(cell, unitcell_sim, dim, kpt, QEType.Relax, usePrimitiveCell=pcell )

cores=os.cpu_count()//2

if os.system(f'mpirun -np {cores} pw.x -nk {cores//4} -inp {unitcell_sim}' ):
    raise IOError("pw.x fail")
