import ctypes
import numpy as np
import glob
import os
import time

def getPromptLib():
    pxpath = os.getenv('PTPATH')
    if pxpath is None:
        raise IOError('PTPATH enviroment is not set')
    libfile = glob.glob(pxpath +'/promptbin/libTak.so')[0]
    taklib = ctypes.CDLL(libfile)
    return taklib
