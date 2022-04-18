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

    def plot(self):
        try:
            import matplotlib.pyplot as plt
            edge = self.getEdge()
            center = edge[:-1]+np.diff(edge)*0.5
            plt.plot(center, self.getWeight())

        except Exception as e:
            print (e)
            print (sys.exc_type)



# Prompt::Hist2D
_pt_Hist2D_new = importFunc('pt_Hist2D_new', type_voidp, [type_dbl, type_dbl, type_uint, type_dbl, type_dbl, type_uint])
_pt_Hist2D_delete = importFunc('pt_Hist2D_delete', None, [type_voidp])
_pt_Hist2D_getWeight = importFunc('pt_Hist2D_getWeight', None, [type_voidp, type_npdbl2d])
_pt_Hist2D_fill = importFunc('pt_Hist2D_fill', None, [type_voidp, type_dbl, type_dbl, type_dbl])
_pt_Hist2D_fill_many = importFunc('pt_Hist2D_fillmany', None, [type_voidp, type_sizet, type_npdbl1d, type_npdbl1d, type_npdbl1d])

class Hist2D():
    def __init__(self, xmin, xmax, xnum, ymin, ymax, ynum):
        self.cobj = _pt_Hist2D_new(xmin, xmax, xnum, ymin, ymax, ynum)
        self.xedge = np.linspace(xmin, xmax, xnum+1)
        self.xcenter = self.xedge[:-1]+np.diff(self.xedge)*0.5

        self.yedge = np.linspace(ymin, ymax, ynum+1)
        self.ycenter = self.yedge[:-1]+np.diff(self.yedge)*0.5

        self.xNumBin = xnum
        self.yNumBin = ynum

    def __del__(self):
        _pt_Hist2D_delete(self.cobj)

    def getEdge(self):
        return self.xedge, self.yedge

    def getWeight(self):
        w = np.zeros([self.xNumBin, self.yNumBin])
        _pt_Hist2D_getWeight(self.cobj, w)
        return w

    def fill(self, x, y, weight=1.):
        _pt_Hist2D_fill(self.cobj, x, y, weight)

    def fillmany(self, x, y, weight=None):
        if weight is None:
            weight = np.ones(x.size)
        if x.size !=weight.size and x.size !=y.size:
            raise RunTimeError('fillnamy different size')
        _pt_Hist2D_fill_many(self.cobj, x.size, x, y, weight )

    def plot(self):
        try:
            import matplotlib.pyplot as plt
            import matplotlib.colors as colors
            fig=plt.figure()
            ax = fig.add_subplot(111)
            H = self.getWeight().T

            X, Y = np.meshgrid(self.xcenter, self.ycenter)
            pcm = ax.pcolormesh(X, Y, H, cmap=plt.cm.jet, shading='auto')
            fig.colorbar(pcm, ax=ax)
            plt.grid()

        except Exception as e:
            print(e)
