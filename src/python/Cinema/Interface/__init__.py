#!/usr/bin/env python3

################################################################################
##                                                                            ##
##  This file is part of Prompt (see https://gitlab.com/xxcai1/Prompt)        ##
##                                                                            ##
##  Copyright 2021-2024 Prompt developers                                     ##
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
import pathlib

type_sizet, type_sizetp = (ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t))
type_bool = ctypes.c_bool
type_int = ctypes.c_int
type_intp = ctypes.POINTER(ctypes.c_int)
type_uint = ctypes.c_uint
type_uintp = ctypes.POINTER(ctypes.c_uint)
type_dbl = ctypes.c_double
type_dblp = ctypes.POINTER(ctypes.c_double)
type_cstr = ctypes.c_char_p
type_voidp = ctypes.c_void_p
type_pyobject = ctypes.py_object

type_npdbl1d=np.ctypeslib.ndpointer(dtype=np.float64,ndim=1,flags='C_CONTIGUOUS')
type_npdbl2d=np.ctypeslib.ndpointer(dtype=np.float64,ndim=2,flags='C_CONTIGUOUS')
type_npsbl2d=np.ctypeslib.ndpointer(dtype=np.float32,ndim=2,flags='C_CONTIGUOUS')
type_npdbl3d=np.ctypeslib.ndpointer(dtype=np.float64,ndim=3,flags='C_CONTIGUOUS')
type_npszt1d=np.ctypeslib.ndpointer(dtype=ctypes.c_size_t,ndim=1,flags='C_CONTIGUOUS')
type_npuint1d=np.ctypeslib.ndpointer(dtype=np.uintc,ndim=1,flags='C_CONTIGUOUS')
type_npcplx2d=np.ctypeslib.ndpointer(dtype=np.complex128,ndim=2,flags='C_CONTIGUOUS')

type_npint641d=np.ctypeslib.ndpointer(dtype=np.int64,ndim=1,flags='C_CONTIGUOUS')



def _getPromptLib():
    _ptpath = os.getenv('CINEMAPATH')
    try:
        libfile = glob.glob(_ptpath +'/cinemabin/src/cxx/libprompt_core.so')[0]
    except:
        try:
            libfile = glob.glob(_ptpath + os.sep + 'libprompt_core.so')[0]
        except:
            raise IOError('CINEMAPATH enviroment need to be configured')
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

def singleton(cls):
    _instance = {}

    def inner():
        if cls not in _instance:
            _instance[cls] = cls()
        return _instance[cls]
    return inner


try:
    import gvar as gv
except ImportError:
    print("The 'gvar' library is not installed. You can install it using pip or conda.")
    print("To install using pip, run: pip install gvar")
    print("To install using conda, run: conda install -c conda-forge gvar")
from scipy.interpolate import interp1d  # CinemaXY


import numpy as np
import gvar as gv
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

class ArrayCoreMixin:
    """Core array functionality mixin"""
    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        return obj
        
    def __array_finalize__(self, obj):
        if obj is None: return
        self.gvar = getattr(obj, 'gvar', None)
        
    def __array_wrap__(self, out_arr, context=None):
        """Ensure mathematical operations preserve attributes"""
        if isinstance(out_arr, np.ndarray) and not isinstance(out_arr, type(self)):
            out_arr = out_arr.view(type(self))
            # Copy all custom attributes
            for name in getattr(self, '_custom_attrs', []):
                setattr(out_arr, name, getattr(self, name, None))
        return out_arr

class ArrayStatsMixin:
    """Statistical operations mixin"""
    @property
    def mean(self):
        if getattr(self, 'gvar', None):
            return np.vectorize(lambda x: x.mean if isinstance(x, gv.GVar) else x)(self)
        return self
        
    @property 
    def sdev(self):
        if getattr(self, 'gvar', None):
            return np.vectorize(lambda x: x.sdev if isinstance(x, gv.GVar) else 0)(self)
        return np.zeros_like(self)
        
    @classmethod
    def from_sdev(cls, mean, sdev=None, **kwargs):
        obj = cls(gv.gvar(mean, sdev) if sdev is not None else mean)
        obj.gvar = True
        for k, v in kwargs.items():
            setattr(obj, k, v)
        return obj
        
    @classmethod
    def from_counts(cls, counts, **kwargs):
        obj = cls(gv.gvar(counts, np.sqrt(counts)))
        obj.gvar = True
        for k, v in kwargs.items():
            setattr(obj, k, v)
        return obj

class ArrayCoordinateMixin:
    """Coordinate system mixin"""
    def __init__(self, *args, x=None, **kwargs):
        self.x = np.asarray(x) if x is not None else np.arange(len(self))
        if len(self.x) != len(self):
            raise ValueError("x coordinates must match array length")
        # Register x as a custom attribute to be preserved
        if not hasattr(self, '_custom_attrs'):
            self._custom_attrs = []
        self._custom_attrs.append('x')
            
    def __array_finalize__(self, obj):
        super().__array_finalize__(obj)
        self.x = getattr(obj, 'x', None)
        
    def __getitem__(self, item):
        result = super().__getitem__(item)
        if isinstance(result, type(self)) and hasattr(self, 'x'):
            result.x = self.x[item] if isinstance(item, (int, slice)) else self.x[np.asarray(item)]
        return result

class ArrayPlotMixin:
    """Plotting functionality mixin"""
    def plot(self, ax=None, plot_errors=True, **kwargs):
        ax = ax or plt.gca()
        x = getattr(self, 'x', np.arange(len(self)))
        y = getattr(self, 'mean', np.asarray(self))
        
        plot_kwargs = {
            'marker': kwargs.pop('marker', 'o'),
            'linestyle': kwargs.pop('linestyle', 'none'),
            'capsize': kwargs.pop('capsize', 3),
            **kwargs
        }
        
        if plot_errors and hasattr(self, 'sdev'):
            yerr = getattr(self, 'sdev')
            ax.errorbar(x, y, yerr=yerr,**plot_kwargs)
        else:
            ax.plot(x, y, **plot_kwargs)
        return ax

# Base array class
class CinemaArray(ArrayCoreMixin, ArrayStatsMixin, np.ndarray):
    pass

# Extended class with coordinates and plotting
class CinemaXY(ArrayCoordinateMixin, ArrayPlotMixin, CinemaArray):
    def interpolate(self, new_x, kind='linear'):
        new_x = np.asarray(new_x)
        if getattr(self, 'gvar', None):
            interp_mean = interp1d(self.x, self.mean, kind=kind)(new_x)
            interp_sdev = interp1d(self.x, self.sdev, kind=kind)(new_x)
            return type(self).from_sdev(interp_mean, interp_sdev, x=new_x)
        return type(self)(interp1d(self.x, self, kind=kind)(new_x), x=new_x)