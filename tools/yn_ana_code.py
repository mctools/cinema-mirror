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

class SQAnaHist1D():
    def __init__(self, filePath, seedStart, seedEnd):
        self.filePath = filePath
        self.seedStart = seedStart
        self.seedEnd = seedEnd
    
    def densityHist(self, hist):
        x_hist = hist.getCentre()
        y_hist = hist.getWeight()/np.diff(hist.getEdge())
        return x_hist, y_hist
        
    def sqtrueHist(self, xmin, xmax, numbin, linear=True):
        sqtrue_hist = Hist1D(xmin, xmax, numbin, linear)
        for i in range(self.seedStart, self.seedEnd+1):
            Data = h5py.File(self.filePath+"/ScororNeutronSq_SofQ_He_seed%d.h5"%i,"r")
            qtrue = np.array(Data['qtrue'])
            weight = np.array(Data['weight'])
            sqtrue_hist.fillmany(qtrue, weight)   
        
        qtrue_hist, s_hist = self.densityHist(sqtrue_hist)
        return qtrue_hist, s_hist, sqtrue_hist 

    def sqHist(self, xmin, xmax, numbin, linear=True):
        sq_hist = Hist1D(xmin, xmax, numbin, linear)
        for i in range(self.seedStart, self.seedEnd+1):
            Data = h5py.File(self.filePath+"/ScororNeutronSq_SofQ_He_seed%d.h5"%i,"r")
            q = np.array(Data['q'])
            weight = np.array(Data['weight'])
            sq_hist.fillmany(q, weight)   
        
        q_hist, s_hist = self.densityHist(sq_hist)
        return q_hist, s_hist, sq_hist 

class MultiScatAna(SQAnaHist1D):
    def __init__(self, filePath, seedStart, seedEnd, scatnum_max):
        super().__init__(filePath, seedStart, seedEnd)
        self.scatnum_max = scatnum_max
    
    def sqtrueHist(self, xmin, xmax, numbin, linear=True):
        s_multiscat = np.zeros((self.scatnum_max,numbin), dtype=float)
        for i in range(1, self.scatnum_max+1):
            sqtrue_hist = Hist1D(xmin, xmax, numbin, linear)
            for j in range(self.seedStart, self.seedEnd+1):
                Data = h5py.File(self.filePath+'/ScororNeutronSq_SofQ_He_NumScat%d_seed%d.h5'%(i, j),"r")
                qtrue = np.array(Data['qtrue'])
                weight = np.array(Data['weight'])
                sqtrue_hist.fillmany(qtrue, weight)
            qtrue_hist, s_hist = self.densityHist(sqtrue_hist) 
            s_multiscat[i-1,:] = s_hist
        qtrue_multiscat = qtrue_hist
        return qtrue_multiscat, s_multiscat

    def sqHist(self, xmin, xmax, numbin, linear=True):
        s_multiscat = np.zeros((self.scatnum_max,numbin), dtype=float)
        for i in range(1, self.scatnum_max+1):
            sq_hist = Hist1D(xmin, xmax, numbin, linear)
            for j in range(self.seedStart, self.seedEnd+1):
                Data = h5py.File(self.filePath+'/ScororNeutronSq_SofQ_He_NumScat%d_seed%d.h5'%(i, j),"r")
                q = np.array(Data['q'])
                weight = np.array(Data['weight'])
                sq_hist.fillmany(q, weight)
            q_hist, s_hist = self.densityHist(sq_hist) 
            s_multiscat[i-1,:] = s_hist
        q_multiscat = q_hist
        return q_multiscat, s_multiscat

    def probaMultiScat(self):
        scatnum = np.zeros((self.scatnum_max), dtype=int)
        scatnum_proba = np.zeros(shape=(self.scatnum_max), dtype=float)
        for i in range(1, self.scatnum_max+1):
            scatnum[i-1] = i
            proba = 0
            for j in range(self.seedStart, self.seedEnd+1):
                Data = h5py.File(self.filePath+'/ScororNeutronSq_SofQ_He_NumScat%d_seed%d.h5'%(i, j),"r")
                weight = np.array(Data['weight'])
                proba += np.sum(weight)
            scatnum_proba[i-1] = proba
        return scatnum, scatnum_proba

        
       







