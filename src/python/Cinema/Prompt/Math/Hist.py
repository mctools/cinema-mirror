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

from Cinema.Interface import *
import numpy as np



_pt_Hist1D_new = importFunc('pt_Hist1D_new', type_voidp, [type_dbl, type_dbl, type_uint, type_bool])
_pt_Hist1D_delete = importFunc('pt_Hist1D_delete', None, [type_voidp])
_pt_Hist1D_getEdge = importFunc('pt_Hist1D_getEdge', None, [type_voidp, type_npdbl1d])
_pt_Hist1D_getHit = importFunc('pt_Hist1D_getHit', None, [type_voidp, type_npdbl1d])
_pt_Hist1D_getWeight = importFunc('pt_Hist1D_getWeight', None, [type_voidp, type_npdbl1d])
_pt_Hist1D_fill = importFunc('pt_Hist1D_fill', None, [type_voidp, type_dbl, type_dbl])
_pt_Hist1D_fill_many = importFunc('pt_Hist1D_fillmany', None, [type_voidp, type_sizet, type_npdbl1d, type_npdbl1d])


class Hist1D():
    def __init__(self, xmin, xmax, num, linear=True):
        self.cobj = _pt_Hist1D_new(xmin, xmax, num, linear)
        self.numbin = num

    def __del__(self):
        _pt_Hist1D_delete(self.cobj)

    def getEdge(self):
        edge = np.zeros(self.numbin+1)
        _pt_Hist1D_getEdge(self.cobj, edge)
        return edge

    def getHit(self):
        hit = np.zeros(self.numbin)
        _pt_Hist1D_getHit(self.cobj, hit)
        return hit

    def getCentre(self):
        edge = self.getEdge()
        center = edge[:-1]+np.diff(edge)*0.5
        return center

    def getWeight(self):
        w = np.zeros(self.numbin)
        _pt_Hist1D_getWeight(self.cobj, w)
        return w

    def fill(self, x, weight=1.):
        _pt_Hist1D_fill(self.cobj, x, weight)

    def fillmany(self, x, weight=None):
        if weight is None:
            weight = np.ones(x.size)
        if(x.size !=weight.size):
            raise RunTimeError('fillnamy different size')
        _pt_Hist1D_fill_many(self.cobj, x.size, x, weight )

    def plot(self, show=False, label=None):
        try:
            import matplotlib.pyplot as plt
            center = self.getCentre()
            w = self.getWeight()
            uncet = np.sqrt(self.getHit()/10.)
            err = np.divide(w, uncet, where=(uncet!=0.))
            plt.errorbar(center, w, yerr=err, fmt='o', label=label)

            if show:
                plt.show()
        except Exception as e:
            print (e)


_pt_Est1D_new = importFunc('pt_Est1D_new', type_voidp, [type_dbl, type_dbl, type_uint, type_bool])
_pt_Est1D_delete = importFunc('pt_Est1D_delete', None, [type_voidp])
_pt_Est1D_fill = importFunc('pt_Est1D_fill', None, [type_voidp, type_dbl, type_dbl, type_dbl])

class Est1D(Hist1D):
    def __init__(self, xmin, xmax, num, linear=True):
        self.cobj = _pt_Est1D_new(xmin, xmax, num, linear)
        self.numbin = num

    def __del__(self):
        _pt_Est1D_delete(self.cobj)

    def fill(self, x, w, e):
        _pt_Est1D_fill(self.cobj, x, w, e)

    def fillmany(self, x, w, e):
        vfillxwh = np.vectorize(self.fill)
        return vfillxwh(x, w, e)

    def getError(self):
        return self.getHit() #hit contains error in this class

    def plot(self, show=False, label=None):
        try:
            import matplotlib.pyplot as plt
            center = self.getCentre()
            w = self.getWeight()
            err = self.getError()
            plt.errorbar(center, w, yerr=err, fmt='o', label=label)
            if show:
                plt.show()
        except Exception as e:
            print (e)

class SpectrumEstimator(Hist1D):
    def __init__(self, xmin, xmax, num, linear=True):
        super().__init__(xmin, xmax, num, linear)
        self.hitCounter = Hist1D(xmin, xmax, num, linear)

    def fill(self, x, weight, hit):
        super().fill(x, weight)
        self.hitCounter.fill(x, hit)

    def fillmany(self, x, weight, hit):
        vfillxwh = np.vectorize(self.fill)
        return vfillxwh(x, weight, hit)

    def getError(self):
        uncet = np.sqrt(self.hitCounter.getHit()/10.)
        err = np.divide(self.getWeight(), uncet, where=(uncet!=0.))
        return err

    def plot(self, show=False, label=None):
        try:
            import matplotlib.pyplot as plt
            center = self.getCentre()
            w = self.getWeight()
            err = self.getError()
            plt.errorbar(center, w, yerr=err, fmt='o', label=label)
            if show:
                plt.show()
        except Exception as e:
            print (e)




# Prompt::Hist2D
_pt_Hist2D_new = importFunc('pt_Hist2D_new', type_voidp, [type_dbl, type_dbl, type_uint, type_dbl, type_dbl, type_uint])
_pt_Hist2D_delete = importFunc('pt_Hist2D_delete', None, [type_voidp])
_pt_Hist2D_getWeight = importFunc('pt_Hist2D_getWeight', None, [type_voidp, type_npdbl2d])
_pt_Hist2D_getHit = importFunc('pt_Hist2D_getHit', None, [type_voidp, type_npdbl2d])
_pt_Hist2D_getDensity = importFunc('pt_Hist2D_getDensity', None, [type_voidp, type_npdbl2d])
_pt_Hist2D_fill = importFunc('pt_Hist2D_fill', None, [type_voidp, type_dbl, type_dbl, type_dbl])
_pt_Hist2D_fill_many = importFunc('pt_Hist2D_fillmany', None, [type_voidp, type_sizet, type_npdbl1d, type_npdbl1d, type_npdbl1d])

class Hist2D():
    def __init__(self, xmin, xmax, xnum, ymin, ymax, ynum, metadata):
        self.cobj = _pt_Hist2D_new(xmin, xmax, xnum, ymin, ymax, ynum)
        self.xedge = np.linspace(xmin, xmax, xnum+1)
        self.xcenter = self.xedge[:-1]+np.diff(self.xedge)*0.5

        self.yedge = np.linspace(ymin, ymax, ynum+1)
        self.ycenter = self.yedge[:-1]+np.diff(self.yedge)*0.5

        self.xNumBin = xnum
        self.yNumBin = ynum

        self.xmin = xmin
        self.xmax = xmax

        self.ymin = ymin
        self.ymax = ymax

        self.metadata = metadata

    def __del__(self):
        _pt_Hist2D_delete(self.cobj)

    def getEdge(self):
        return self.xedge, self.yedge

    def getWeight(self):
        w = np.zeros([self.xNumBin, self.yNumBin])
        _pt_Hist2D_getWeight(self.cobj, w)
        return w

    def getHit(self):
        hit = np.zeros([self.xNumBin,self.yNumBin])
        _pt_Hist2D_getHit(self.cobj, hit)
        return hit

    def getDensity(self):
        d = np.zeros([self.xNumBin, self.yNumBin])
        _pt_Hist2D_getWeight(self.cobj, d)
        return d

    def fill(self, x, y, weight=1.):
        _pt_Hist2D_fill(self.cobj, x, y, weight)

    def fillmany(self, x, y, weight=None):
        if weight is None:
            weight = np.ones(x.size)
        if x.size !=weight.size and x.size !=y.size:
            raise RunTimeError('fillnamy different size')
        _pt_Hist2D_fill_many(self.cobj, x.size, x, y, weight )

    def plot(self, show=False):
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
            if show:
                plt.show()

        except Exception as e:
            print(e)

    def save(self, fn):
        import h5py
        f0=h5py.File(fn,"w")
        f0.create_dataset("q", data=self.xcenter, compression="gzip")
        f0.create_dataset("omega", data=self.ycenter, compression="gzip")
        X, Y = np.meshgrid(np.diff(self.yedge), np.diff(self.xedge))
        f0.create_dataset("s", data=self.getWeight()/(X*Y), compression="gzip")
        mtd = f0.create_group('metadata')
        for key, value in self.metadata.items():
            mtd.create_dataset(key, data = value)
        f0.close()



# Class NumpyHist2D is written to validate the class Hist2D only. It shouldn't be used in practice due to its significantly slow performance.
class NumpyHist2D():
    def __init__(self, xbin, ybin, range):
        range=np.array(range)
        if range.shape != (2,2):
            raise IOError('wrong range shape')
        self.range=range
        self.xedge=np.linspace(range[0][0], range[0][1], xbin+1)
        self.yedge=np.linspace(range[1][0], range[1][1], ybin+1)
        if range[0][0] == range[0][1] or range[1][0] == range[1][1]:
            raise IOError('wrong range input')
        self.xbinfactor=xbin/float(range[0][1]-range[0][0])
        self.ybinfactor=ybin/float(range[1][1]-range[1][0])
        self.xmin=range[0][0]
        self.xmax=range[0][1]
        self.ymin=range[1][0]
        self.ymax=range[1][1]
        self.hist =np.zeros([xbin, ybin])

    def fill(self, x, y, weights=None):
        h, xedge, yedge = np.histogram2d(x, y, bins=[self.xedge, self.yedge], weights=weights)
        self.hist += h

    def getHistVal(self):
        return self.hist

    def getXedges(self):
        return self.xedge

    def getYedges(self):
        return self.yedge

    def show(self):
        import matplotlib.pyplot as plt
        fig=plt.figure()
        ax = fig.add_subplot(111)
        H = self.hist.T

        X, Y = np.meshgrid(self.xedge, self.yedge)
        import matplotlib.colors as colors
        pcm = ax.pcolormesh(X, Y, H, cmap=plt.cm.jet,  norm=colors.LogNorm(vmin=H.max()*1e-4, vmax=H.max()),)
        fig.colorbar(pcm, ax=ax)
        plt.xlabel('Q, Aa^-1')
        plt.ylabel('energy, eV')
        plt.show()
