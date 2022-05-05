#!/usr/bin/env python3

import os,glob, getopt

fs=glob.glob('supercell-*.in')
fs=sorted(fs)
cores=os.cpu_count()//2
for f in fs:
    if os.system(f'mpirun -np {cores} pw.x -nk {cores//4} -inp {f} | tee {f[0:-3]}.out' ):
        raise IOError("pw.x fail")

if os.system('phonopy -f supercell-*.out' ):
    raise IOError("force fail")

# if os.system(f'phonopy --qe -c unit_cell.in --dim {dim[0]} {dim[1]} {dim[2]}  --mesh {mesh[0]} {mesh[1]} {mesh[2]} --hdf5-compression gzip --hdf5  --eigvecs --nomeshsym'):
#     raise IOError("mesh fail")


# if os.system('phonopy --qe -d -v --dim="{dim1} {dim2} {dim3}" '.format(dim1=dim[0],dim2=dim[1],dim3=dim[2])+ " -c " + simname):
#     raise IOError("pw.x fail")
