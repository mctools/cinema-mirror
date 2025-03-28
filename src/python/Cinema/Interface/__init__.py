#!/usr/bin/env python3

################################################################################
##                                                                            ##
##  This file is part of Prompt (see https://gitlab.com/xxcai1/Prompt)        ##
##                                                                            ##
##  Copyright 2021-2022 Prompt developers                                     ##
##                                                                            ##
##  Licensed under the Apache License, Version 2.0 (the "License");           ##
##  you may not use this file except in compliance with the License.          ##
##  You may obtain a copy of the License at                                   ##
##                                                                            ##
##      http://www.apache.org/licenses/LICENSE-2.0                            ##
##                                                                            ##
##  Unless required by applicable law or agreed to in writing, software       ##
##  distributed under the License is distributed on an "AS IS" BASIS,         ##
##  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  ##
##  See the License for the specific language governing permissions and       ##
##  limitations under the License.                                            ##
##                                                                            ##
################################################################################

import ctypes
import numpy as np
import glob
import os

type_sizet, type_sizetp = (ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t))
type_bool = ctypes.c_bool
type_int, type_intp, type_uint, type_uintp, type_dbl, _dblp, type_cstr, type_voidp = (ctypes.c_int, ctypes.POINTER(ctypes.c_int),
                                                   ctypes.c_uint,ctypes.POINTER(ctypes.c_uint), ctypes.c_double,
                                                   ctypes.POINTER(ctypes.c_double), ctypes.c_char_p, ctypes.c_void_p)
type_npdbl1d=np.ctypeslib.ndpointer(dtype=np.float64,ndim=1,flags='C_CONTIGUOUS')
type_npdbl2d=np.ctypeslib.ndpointer(dtype=np.float64,ndim=2,flags='C_CONTIGUOUS')
type_npsbl2d=np.ctypeslib.ndpointer(dtype=np.float32,ndim=2,flags='C_CONTIGUOUS')
type_npdbl3d=np.ctypeslib.ndpointer(dtype=np.float64,ndim=3,flags='C_CONTIGUOUS')
type_npszt1d=np.ctypeslib.ndpointer(dtype=ctypes.c_size_t,ndim=1,flags='C_CONTIGUOUS')
type_npuint1d=np.ctypeslib.ndpointer(dtype=np.uintc,ndim=1,flags='C_CONTIGUOUS')
type_npcplx2d=np.ctypeslib.ndpointer(dtype=np.complex128,ndim=2,flags='C_CONTIGUOUS')


def _getPromptLib():
    _ptpath = os.getenv('CINEMAPATH')
    if _ptpath is None:
        raise IOError('CINEMAPATH enviroment is not set')
    libfile = glob.glob(_ptpath +'/cinemabin/src/cxx/libprompt_core.so')[0]
    return ctypes.CDLL(libfile), _ptpath

_taklib, _ptpath = _getPromptLib()

def _findData(fileName):
    pass

def importFunc(funcName, resType, argType):
    func = getattr(_taklib, funcName)
    func.restype = resType
    func.argtypes = argType
    return func

def plotStyle(fontSize=16):
    import matplotlib.style
    import matplotlib, sys
    import matplotlib.pyplot as plt

    plt.rcParams.update({'font.size': fontSize})
    matplotlib.rcParams['lines.linewidth'] = 2


    _plt_legend_orig = plt.legend
    def _plt_legend(*args,**kwargs):
        notouchframelw = False
        if 'notouchframelw' in kwargs:
            notouchframelw = True
            del kwargs['notouchframelw']
        l = _plt_legend_orig(*args,**kwargs)
        if not notouchframelw:
            l.get_frame().set_linewidth(0.0)
        return l
    plt.legend=_plt_legend
