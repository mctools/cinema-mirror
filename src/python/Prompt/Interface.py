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
type_npdbl3d=np.ctypeslib.ndpointer(dtype=np.float64,ndim=3,flags='C_CONTIGUOUS')
type_npszt1d=np.ctypeslib.ndpointer(dtype=ctypes.c_size_t,ndim=1,flags='C_CONTIGUOUS')

def _getPromptLib():
    _ptpath = os.getenv('PTPATH')
    if _ptpath is None:
        raise IOError('PTPATH enviroment is not set')
    libfile = glob.glob(_ptpath +'/promptbin/src/cxx/libprompt_core.so')[0]
    return ctypes.CDLL(libfile), _ptpath

taklib, ptpath = _getPromptLib()

def _findData(fileName):
    pass

def importFunc(funcName, resType, argType):
    func = getattr(taklib, funcName)
    func.restype = resType
    func.argtypes = argType
    return func
