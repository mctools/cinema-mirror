#!/usr/bin/env python3

import os,glob, getopt

dim=[3,3,3]
mesh=[40,40,40]

#density of states
if os.system(f'phonopy -v --qe -c unit_cell.in --dim {dim[0]} {dim[1]} {dim[2]} --pdos AUTO --mesh {mesh[0]} {mesh[1]} {mesh[2]}   --nowritemesh -p '):
    raise IOError("dos fail")

#mesh
if os.system(f'phonopy --qe -c unit_cell.in --dim {dim[0]} {dim[1]} {dim[2]}  --mesh {mesh[0]} {mesh[1]} {mesh[2]} --hdf5-compression gzip --hdf5  --eigvecs --nomeshsym'):
    raise IOError("mesh fail")

#qpoint
if os.system(f'phonopy -v --qe -c unit_cell.in --dim 3 3 2 --eigvecs --read-qpoints --hdf5'):
    raise IOError("qpoint fail")

if __name__ == '__main__':
    opts, args = getopt.getopt(sys.argv[1:],"m:d:h")
    for o, a in opts:
        if o == '-i':i:k:p:md:
            input = a
        if o == '-k':
            dim_strs=a.split(' ')
            if len(dim_strs) == 1:
                num=int(a)
                kpt[0]=num
                kpt[1]=num
                kpt[2]=num
            else:
                print("k-point number is not an integer. Trying to interpret as an array")
                if len(dim_strs) != 3:
                   raise IOError('k-point given is not an 3-element array')
                kpt[0]=int(dim_strs[0])
                kpt[1]=int(dim_strs[1])
                kpt[2]=int(dim_strs[2])
        if o == '-p':
            if a=='y':
                pcell=True
            elif a=='n':
                pcell=False
            elif a!='n':
                raise IOError('-p should be either "y" or "n"')
        if o == '-m':
            ismetal=True
        elif o == '-d':
            dim_strs=a.split(' ')
            if len(dim_strs) != 3:
                raise IOError('supercell should be in 3D')
            dim[0]=int(dim_strs[0])
            dim[1]=int(dim_strs[1])
            dim[2]=int(dim_strs[2])
        elif o == '-h':
            print ('p: build up supercell by primitive cells. default is "y".')
            print ('d: supercell dimension. default is "2 2 2".')
            print ('k: k-point mesh size. could be a integer or three integers in quotation mark. default is 4.')
            print ('m: material is metal. ')
            print ('i: input POSCAR file')
            sys.exit()
