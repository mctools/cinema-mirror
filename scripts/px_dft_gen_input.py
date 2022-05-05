#!/usr/bin/env python3

import numpy as np
from Cinema.PiXiu import getAtomMassBC, Pseudo
import os,sys, getopt

if __name__ == '__main__':
    input='Use -i option to define input'
    kpt=[4,4,4]
    dim=[2,2,2]
    unitcell_sim="POSCAR"
    pcell = True
    ismetal = False

    opts, args = getopt.getopt(sys.argv[1:],"i:k:p:md:h")
    for o, a in opts:
        if o == '-i':
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
    ps = Pseudo()
    print('pcell is ', pcell)
    ps.qems(input, unitcell_sim, dim, kpt, usePrimitiveCell=pcell, isMetal=ismetal )
