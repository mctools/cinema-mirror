import numpy as np
from Cinema.Interface.units import avogadro
from Cinema.Prompt import PromptFileReader
from Cinema.Interface import plotStyle
import matplotlib.pyplot as plt
import h5py
from glob import glob
import os
import re

class McplAnalysor1D(PromptFileReader):
    def __init__(self,filePath):
        self.filePath = filePath
        
    def filesMany(self):
        path = os.path.join(self.filePath)
        files = glob(path)
        files.sort(key=lambda l: int(re.findall('\d+', l)[-1]))
        return files

    def getHist(self):
        super().__init__(self.filePath)
        edge = self.getData('edge')
        content = self.getData('content')
        return edge[:-1], content/np.diff(edge)
    
    def getHistMany(self, seedStart, seedEnd):
        files = self.filesMany()
        ysum_hist = 0
        for i in range(seedStart-1, seedEnd):
            self.filePath = files[i]
            print(self.filePath)
            x_hist, y_hist = self.getHist()
            ysum_hist += y_hist
        return x_hist, ysum_hist


# constant
inc_d = 7.64 # barn/atom , bound incoherent scattering cross section of deuterium atom
inc_o = 4.232 # barn/atom , bound incoherent scattering cross section of oxygen atom
self_cross_section = (2*inc_d+inc_o)/(4*np.pi)/3  #  barn/atom, self cross section per atom of heavy water
numPar1 = 2e10 # incident neutron number of non-biased run
numPar2 = 2e9 # incident neutron number of biased run

# Soper Data 
soperData=h5py.File("HeavyWater_Soper.h5","r")
q_soper=np.array(soperData['Q'])
d2o_soper=np.array(soperData['d2o_cross_section']) # Interference differential scattering cross section for heavy water
soperData.close()

# data for non-biased run 
HW = McplAnalysor1D('./HW_nonbiased_Data/*ScorerNeutronSq_PofQ_HW_seed*.mcpl.gz')
q_HW, p_HW = HW.getHistMany(seedStart=1, seedEnd=25)

HW1 = McplAnalysor1D('./HW_nonbiased_Data/*ScorerNeutronSq_PofQ1_HW_seed*.mcpl.gz')
q_HW1, p_HW1 = HW1.getHistMany(seedStart=1, seedEnd=25)

HW2 = McplAnalysor1D('./HW_nonbiased_Data/*ScorerNeutronSq_PofQ2_HW_seed*.mcpl.gz')
q_HW2, p_HW2 = HW2.getHistMany(seedStart=1, seedEnd=25)

HW4 = McplAnalysor1D('./HW_nonbiased_Data/*ScorerNeutronSq_PofQ4_HW_seed*.mcpl.gz')
q_HW4, p_HW4 = HW4.getHistMany(seedStart=1, seedEnd=25)

# data for biased run 
HWb = McplAnalysor1D('./HW_biased_Data/*ScorerNeutronSq_PofQ_HW_seed*.mcpl.gz')
q_HWb, p_HWb = HWb.getHistMany(seedStart=1, seedEnd=25)

HWb1 = McplAnalysor1D('./HW_biased_Data/*ScorerNeutronSq_PofQ1_HW_seed*.mcpl.gz')
q_HWb1, p_HWb1 = HWb1.getHistMany(seedStart=1, seedEnd=25)

HWb2 = McplAnalysor1D('./HW_biased_Data/*ScorerNeutronSq_PofQ2_HW_seed*.mcpl.gz')
q_HWb2, p_HWb2 = HWb2.getHistMany(seedStart=1, seedEnd=25)

HWb4 = McplAnalysor1D('./HW_biased_Data/*ScorerNeutronSq_PofQ4_HW_seed*.mcpl.gz')
q_HWb4, p_HWb4 = HWb4.getHistMany(seedStart=1, seedEnd=25)

plotStyle()

# Normalization of simulation data and measured data (Figure 8)
# d2o_soper = d2o_soper+self_cross_section
# plt.scatter(q_soper, d2o_soper/np.trapz(d2o_soper, q_soper), s=12, color='black',label='Measured') #Soper(2013)
# area = np.trapz(p_HW1[0:300], q_HW1[0:300])
# plt.plot(q_HW1, p_HW1/area, label='Simulation') #Pn=1(Q)
# plt.xlabel('Q, Å$^{-1}$') 
# plt.ylabel('Interference DCS, arb.unit')
# plt.grid()
# plt.legend(fontsize=12*1.5, loc='best')
# plt.tight_layout()
# plt.show()

# Comparison between P(Q) for different scattering numbers obtained from non-biased and biased runs (Figure 9)
plt.ticklabel_format(axis='y', style='sci', scilimits=(0,1))
plt.plot(q_HW, p_HW/numPar1*2, zorder=2, label='All Scatterings$\\times$2')
plt.plot(q_HW1, p_HW1/numPar1, zorder=2, label=f'Single Scattering')
plt.plot(q_HW2[0:501:6], p_HW2[0:501:6]/numPar1, zorder=2, label=f'Two Scatterings ')
plt.plot(q_HW4[0:501:6], p_HW4[0:501:6]/numPar1, zorder=2, label=f'Four Scatterings ')
plt.scatter(q_HWb, p_HWb/numPar2*2, marker='s', s=36, c='#e377c2', zorder=1, alpha=0.9)
plt.scatter(q_HWb1, p_HWb1/numPar2, marker='o', s=34, c='#7f7f7f', zorder=1, alpha=0.9)
plt.scatter(q_HWb2[0:501:7], p_HWb2[0:501:7]/numPar2, marker='d', s=32, c='#9467bd', zorder=1, alpha=0.9)
plt.scatter(q_HWb4[0:501:7], p_HWb4[0:501:7]/numPar2, marker='>', s=32, c='#17becf', zorder=1, alpha=0.9)
plt.xlabel('Q, Å$^{-1}$') 
plt.ylabel('P(Q), Å/neutron')
plt.legend(fontsize=13, ncol=2, loc='best')
plt.tight_layout()
plt.show()
