#!/usr/bin/env python3

import ctypes
import numpy as np
import glob
import os
import time

npf1d=np.ctypeslib.ndpointer(dtype=np.float64,ndim=1,flags='C_CONTIGUOUS')
npf2d=np.ctypeslib.ndpointer(dtype=np.float64,ndim=2,flags='C_CONTIGUOUS')
npf3d=np.ctypeslib.ndpointer(dtype=np.float64,ndim=3,flags='C_CONTIGUOUS')


def getPromptLib():
    _ptpath = os.getenv('PTPATH')
    if _ptpath is None:
        raise IOError('PTPATH enviroment is not set')
    libfile = glob.glob(_ptpath +'/promptbin/src/cxx/libprompt_core.so')[0]
    return ctypes.CDLL(libfile), _ptpath

_taklib, _ptpath = getPromptLib()

def findData(fileName):
    pass

class Launcher():
    _taklib.pt_Launcher_getInstance.restype = ctypes.c_void_p
    _taklib.pt_Launcher_getInstance.argtypes = []
    def __init__(self):
        self.cobj = _taklib.pt_Launcher_getInstance()

    _taklib.pt_Launcher_setSeed.restype = None
    _taklib.pt_Launcher_setSeed.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    def setSeed(seed):
        _taklib.pt_Launcher_setSeed(self.cobj, seed)

    _taklib.pt_Launcher_loadGeometry.restype = None
    _taklib.pt_Launcher_loadGeometry.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    def loadGeometry(self, fileName):
        _taklib.pt_Launcher_loadGeometry(self.cobj, fileName.encode('utf-8'));
