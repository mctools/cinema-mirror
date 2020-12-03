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
        h, xedge, yedge = np.histogram2d(x, y, bins=[self.xedge, self.yedge], weights=weights)
        self.hist += h

    def getHistVal(self):
        return self.hist
