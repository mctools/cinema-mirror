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

from Interface import *
import numpy as np

_pt_Hist1D_new = importFunc('pt_Hist1D_new', type_voidp, [type_dbl, type_dbl, type_uint, type_bool])
_pt_Hist1D_delete = importFunc('pt_Hist1D_delete', None, [type_voidp])
_pt_Hist1D_getEdge = importFunc('pt_Hist1D_getEdge', None, [type_voidp, type_npdbl1d])
_pt_Hist1D_getWeight = importFunc('pt_Hist1D_getWeight', None, [type_voidp, type_npdbl1d])
_pt_Hist1D_fill = importFunc('pt_Hist1D_fill', None, [type_voidp, type_dbl, type_dbl])
_pt_Hist1D_fill_many = importFunc('pt_Hist1D_fillmany', None, [type_voidp, type_sizet, type_npdbl1d, type_npdbl1d])


class Hist1D():
    def __init__(self, xmin, ylim, num, linear=True):
        self.cobj = _pt_Hist1D_new(xmin, ylim, num, linear)
        self.numbin = num

    def __del__(self):
        _pt_Hist1D_delete(self.cobj)

    def getEdge(self):
        edge = np.zeros(self.numbin+1)
        _pt_Hist1D_getEdge(self.cobj, edge)
        return edge

    def getWeight(self):
        w = np.zeros(self.numbin)
        _pt_Hist1D_getWeight(self.cobj, w)
        return w

    def fill(self, x, weight):
        _pt_Hist1D_fill(self.cobj, x, weight)

    def fillmany(self, x, weight):
        if(x.size !=weight.size):
            raise RunTimeError('fillnamy different size')
        _pt_Hist1D_fill_many(self.cobj, x.size, x, weight )
