
__all__ = ['PromptFileReader']

import mcpl
from io import BytesIO
from glob import glob
import os
import re
import numpy as np
from Cinema.Experiment.Analyser import ErrorPropagator

_recodedict = {}
_recodedict['ekin'] = 'ekin'
_recodedict['q'] = 'polx'
_recodedict['qtrue'] = 'poly'
_recodedict['ekin_atbirth'] = 'polz'
_recodedict['ekin_tof'] = 'x'
# _recodedict['dummy1'] = 'y'
# _recodedict['dummy1'] = 'z'
# _recodedict['dummyvector1'] = 'ux'
# _recodedict['dummyvector2'] = 'uy'
# _recodedict['dummyvector3'] = 'uz'
_recodedict['time'] = 'time'
_recodedict['weight'] = 'weight'
_recodedict['scatNum'] = 'pdgcode'
# _recodedict['dummy3'] = 'userflags'

class PromptFileReader:
    def __init__(self, fn, particleBlocklength=10000, dumpHeader=True):
        self.pfile = mcpl.MCPLFile(fn)
        self.particleBlocklength = particleBlocklength
        if dumpHeader:
            self.pfile.dump_hdr()
            print("comments:\n", self.getComments())

    def dataKeys(self):
        return self.pfile.blobs.keys()

    def getData(self, k):
        raw=BytesIO(self.pfile.blobs[k])
        return np.load(raw)

    def getComments(self):
        return self.pfile.comments

    # this can be used like:
    # for p in reader.blockIterator():
    #     print( p.x, p.y, p.z, p.ekin )
    def blockIterator(self):
     return self.pfile.particle_blocks

    def particleIterator(self):
     return self.pfile.particle_blocks

    def getRecordKeys(self):
        return _recodedict.keys()

    def getRecordData(self, pb, recordkey):
        value = getattr(pb, _recodedict[recordkey])
        return value
    
class McplAnalysor(PromptFileReader):
    def __init__(self, filePath, particleBlocklength=10000, dumpHeader=True):
        self.filePath = filePath
        self.particleBlocklength = particleBlocklength
        self.dumpHeader = dumpHeader
        
    def filesMany(self):
        path = os.path.join(self.filePath)
        files = glob(path)
        files.sort(key=lambda l: int(re.findall('\d+', l)[-1]))
        return files

    def getHist(self):
        super().__init__(self.filePath, particleBlocklength=self.particleBlocklength, dumpHeader=self.dumpHeader)
        content = self.getData('content')
        hit = self.getData('hit')
        if content.ndim == 1:
            edge = self.getData('edge')
            hist = ErrorPropagator(weight=content,  xcentre=edge, count=hit, error=None)
        elif content.ndim == 2:
            xedge = self.getData('xedge')
            yedge = self.getData('yedge')
            hist = ErrorPropagator(weight=content,  xcentre=xedge, ycentre=yedge,  count=hit, error=None)
        
        return hist
    
    def getHistMany(self, offset=0, num=None):
        histMany = None
        files = self.filesMany()
        if num is None:
            num = len(files)
        offset = max(0, offset)
        num = min(num, len(files))
        
        for i in range(offset, num):
            self.filePath = files[i]
            print(self.filePath)
            if histMany is None:
                histMany = self.getHist()
            else:
                histMany += self.getHist()
        
        return histMany

