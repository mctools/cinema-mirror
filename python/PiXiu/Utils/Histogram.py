import numpy as np

class Hist2D():
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
        if (x.max()>= self.xmax) or (y.max()>= self.ymax) or (x.min()<=self.xmin) or  (y.min()<=self.ymin):
            return False
        if x.size != y.size:
            return False

        xi = ((x-self.xmin)*self.xbinfactor).astype(np.int)
        yi = ((y-self.ymin)*self.ybinfactor).astype(np.int)
        if weights is not None:
            self.hist[xi, yi] += weights
        else:
            self.hist[xi, yi] += 1
        return True

    def slowfill(self, x, y, weights=None):
        h, xedge, yedge = np.histogram2d(x, y, bins=[self.xedge, self.yedge], weights=weights)
        self.hist += h
