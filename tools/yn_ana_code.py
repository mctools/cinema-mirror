import numpy as np
from Cinema.Prompt.Math.Hist import Hist1D
from Cinema.Interface.units import avogadro
import h5py

# atom number in material        
def atomNum(d2o_r, d2o_h, v_r, v_h): # mm
    d2o_density = 1.10526/1000 # g/mm3
    d2o_molmass = 2*2+16 # g/mol
    d2o_num = 3
    v_density = 6.11/1000 # g/mm3
    v_molmass = 50.94 # g/mol
    v_num = 1
    d2o_volume=np.pi*np.square(d2o_r)*d2o_h
    v_volume=np.pi*np.square(v_r)*v_h
    d2o_atomNum=d2o_volume*d2o_density/d2o_molmass*avogadro*d2o_num
    v_atomNum=v_volume*v_density/v_molmass*avogadro*v_num
    return d2o_atomNum, v_atomNum

class WgtFileAnalysor(Hist1D):
    def __init__(self, xmin, xmax, numbin, xname, weightname, linear=True):
        self.xmin = xmin
        self.xmax = xmax
        self.numbin = numbin
        self.xname = xname
        self.linear = linear
        self. weightname = weightname

    def densityHist(self):
        x_hist = self.getCentre()
        y_hist = self.getWeight()/np.diff(self.getEdge())
        return x_hist, y_hist
    
    def fillHist(self, filePath, seedStart, seedEnd, scatnum=None):
        super().__init__(self.xmin, self.xmax, self.numbin, self.linear)
        for i in range(seedStart, seedEnd+1):
            Data = h5py.File(filePath+"/ScororNeutronSq_SofQ_He_seed%d.h5"%i,"r")
            x = np.array(Data[self.xname])
            weight = np.array(Data[self.weightname])
            if scatnum:
                numScat = np.array(Data['numScat'])
                idx = np.where(numScat==scatnum)
                x = x[idx]
                weight = weight[idx]
            self.fillmany(x, weight)
        x_hist, y_hist = self.densityHist()
        return x_hist, y_hist

class SQTrueHist(WgtFileAnalysor):
    def __init__(self, xmin, xmax, numbin,linear=True):
        super().__init__(xmin, xmax, numbin, 'qtrue', 'weight', linear)       

class SQHist(WgtFileAnalysor):
    def __init__(self, xmin, xmax, numbin,linear=True):
        super().__init__(xmin, xmax, numbin, 'q', 'weight', linear)
    
class MultiScatAnalysor(WgtFileAnalysor):
    def __init__(self, filePath, seedStart, seedEnd):
        self.filePath = filePath
        self.seedStart = seedStart
        self.seedEnd = seedEnd
        
    def sqHist(self, xmin, xmax, numbin, scatnum, inelastic=True, linear=True):
        if inelastic:
            super().__init__(xmin, xmax, numbin, 'qtrue', 'weight', linear)
        else:
            super().__init__(xmin, xmax, numbin, 'q', 'weight', linear)
        q_hist, s_hist= self.fillHist(self.filePath, self.seedStart, self.seedEnd, scatnum)
        return q_hist, s_hist
    
    def probaMultiScat(self, scatnum):
        proba = 0
        for i in range(self.seedStart, self.seedEnd+1):
            Data = h5py.File(self.filePath+"/ScororNeutronSq_SofQ_He_seed%d.h5"%i,"r")
            numScat = np.array(Data['numScat'])
            idx = np.where(numScat==scatnum)
            weight = np.array(Data['weight'])[idx]
            proba += np.sum(weight)
        return proba
            
            
        






        
       







