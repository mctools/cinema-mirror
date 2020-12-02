import ctypes
import numpy as np
import glob
import os
import time

pxpath = os.getenv('PIXIUPATH')
if pxpath is None:
    raise IOError('PIXIUPATH enviroment is not set')

libfile = glob.glob(pxpath +'/rundir/cxx/src/libPiXiu.so')[0]
pxlib = ctypes.cdll.LoadLibrary(libfile)

npDouble1D = np.ctypeslib.ndpointer(dtype=np.float64,ndim=1,flags='C_CONTIGUOUS')
charptr = ctypes.POINTER(ctypes.c_char)
doubleptr = ctypes.POINTER(ctypes.c_double)

class NumpyHist2D(object):
    pxlib.NumpyHist2D_new.restype = ctypes.c_void_p
    pxlib.NumpyHist2D_new.argtypes = [ctypes.c_uint, ctypes.c_double, ctypes.c_double, ctypes.c_uint, ctypes.c_double, ctypes.c_double ]

    pxlib.NumpyHist2D_delete.restype = None
    pxlib.NumpyHist2D_delete.argtypes = [ctypes.c_void_p]

    pxlib.NumpyHist2D_fill.restype = None
    pxlib.NumpyHist2D_fill.argtypes = [ctypes.c_void_p, ctypes.c_uint, npDouble1D, npDouble1D]

    pxlib.NumpyHist2D_fillWeighted.restype = None
    pxlib.NumpyHist2D_fillWeighted.argtypes = [ctypes.c_void_p, ctypes.c_uint, npDouble1D, npDouble1D, npDouble1D]

    pxlib.NumpyHistBase_save.restype = None
    pxlib.NumpyHistBase_save.argtypes = [ctypes.c_void_p, charptr]

    pxlib.NumpyHistBase_getRaw.restype = doubleptr
    pxlib.NumpyHistBase_getRaw.argtypes = [ctypes.c_void_p]

    pxlib.NumpyHist2D_getNBinX.restype = ctypes.c_uint
    pxlib.NumpyHist2D_getNBinX.argtypes = [ctypes.c_void_p]

    pxlib.NumpyHist2D_getNBinY.restype = ctypes.c_uint
    pxlib.NumpyHist2D_getNBinY.argtypes = [ctypes.c_void_p]

    def __init__(self, xbin, ybin, range):
        self.xmin=range[0][0]
        self.xmax=range[0][1]
        self.ymin=range[1][0]
        self.ymax=range[1][0]
        self.xbin=xbin
        self.ybin=ybin
        self.self = pxlib.NumpyHist2D_new(xbin, range[0][0], range[0][1], ybin, range[1][0], range[1][1])

    def __del__(self):  # when reference count hits 0 in Python,
        pxlib.NumpyHist2D_delete(self.self)  # call C++ vector destructor

    def save(self, fn):
        pxlib.NumpyHistBase_save(self.self, bytes(fn, encoding='utf8') )

    def getNBinX(self):
        return pxlib.NumpyHist2D_getNBinX(self.self)

    def getNBinY(self):
        return pxlib.NumpyHist2D_getNBinY(self.self)

    def getHistVal(self):
        return  np.ctypeslib.as_array(ctypes.cast(pxlib.NumpyHistBase_getRaw(self.self),doubleptr) ,shape=[self.getNBinX(),self.getNBinY()])

    # def __len__(self):
    #     return pxlib.std_vector_size(self.self)
    #
    # def __getitem__(self, i):  # access elements in vector at index
    #     if 0 <= i < len(self):
    #         return pxlib.std_vector_get(self.self, ctypes.c_int(i))
    #     raise IndexError('Vector index out of range')
    #
    # def __repr__(self):
    #     return '[{}]'.format(', '.join(str(self[i]) for i in range(len(self))))

    def fill(self, x, y, weight=None):  # push calls vector's push_back
        if x.size!=y.size:
            raise IOError('histgraom input vector x and y are in different size')
        if weight is None:
            pxlib.NumpyHist2D_fill(self.self, x.size, x, y)
        else:
            if weight.size!=x.size:
                raise IOError('weight has wrong size')
            pxlib.NumpyHist2D_fillWeighted(self.self, x.size, x, y, weight)
