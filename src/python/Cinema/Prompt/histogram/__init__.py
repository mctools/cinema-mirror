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

# __all__ = ['eKin2k', 'angleCosine2Q', 'wl2ekin', 'ekin2wl', 'ekin2v', 'v2ekin', ' angleCosine2QMany', 'v2ekinMany']
__all__ = []
from . import Hist
from .Hist import *
__all__ += Hist.__all__

from Cinema.Interface import *
import numpy as np

eKin2k = importFunc('pt_eKin2k', type_dbl, [type_dbl] )
angleCosine2Q = importFunc('pt_angleCosine2Q', type_dbl, [type_dbl, type_dbl, type_dbl] )
wl2ekin = importFunc('pt_wl2ekin', type_dbl, [type_dbl] )
ekin2wl = importFunc('pt_ekin2wl', type_dbl, [type_dbl] )

#output unit mm/sec
ekin2v = importFunc('pt_ekin2speed', type_dbl, [type_dbl] )
v2ekin = importFunc('pt_speed2ekin', type_dbl, [type_dbl] )

# def elasticQ(cosAngle, fl)
angleCosine2QMany = np.vectorize(angleCosine2Q)
v2ekinMany = np.vectorize(v2ekin)

# def qElastic(l)

try:
    import gvar as gv
except ImportError:
    print("The 'gvar' library is not installed. You can install it using pip or conda.")
    print("To install using pip, run: pip install gvar")
    print("To install using conda, run: conda install -c conda-forge gvar")

class CinemaArray(np.ndarray):
    def __new__(cls, input_array):
        # Convert input to ndarray, then view as CinemaArray
        obj = np.asarray(input_array).view(cls)
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        # Copy attributes from the original object
        self.gvar = getattr(obj, 'gvar', None)

    def __array_wrap__(self, out_arr, context=None):
        # Wrap the output array as CinemaArray
        return np.ndarray.__array_wrap__(self, out_arr, context)

    def __getitem__(self, item):
        result = super(CinemaArray, self).__getitem__(item)
        if isinstance(result, np.ndarray):
            return result.view(CinemaArray)
        else:
            return result

    @property
    def mean(self):
        if self.gvar is not None:
            return np.vectorize(lambda x: x.mean if isinstance(x, gv.GVar) else x)(self)
        return self

    @property
    def sdev(self):
        if self.gvar is not None:
            return np.vectorize(lambda x: x.sdev if isinstance(x, gv.GVar) else 0)(self)
        return np.zeros_like(self)

    @staticmethod
    def from_sdev(mean, sdev=None):
        if sdev is None:
            return CinemaArray(mean)
        else:
            gvars = gv.gvar(mean, sdev)
            obj = np.asarray(gvars).view(CinemaArray)
            obj.gvar = True
            return obj

    @staticmethod
    def from_counts(counts):
        gvars = gv.gvar(counts, counts/np.sqrt(counts))
        obj = np.asarray(gvars).view(CinemaArray)
        obj.gvar = True
        return obj


