import numpy as np
from Cinema.Interface import plotStyle
import matplotlib.pyplot as plt
import h5py
from sq_1dhist_ana import atomNum

# calculate the probability of every scattering number
def probaMultiScat(filePath, seedStart, seedEnd, scatnum_max):
    scatnum = np.empty(shape=[scatnum_max,], dtype=int)
    scatnum_proba = np.empty(shape=[scatnum_max,], dtype=float)
    for i in range(1, scatnum_max+1):
        scatnum[i-1] = i
        proba = 0
        for j in range(seedStart, seedEnd+1):
            Data = h5py.File(filePath+'/ScororNeutronSq_SofQ_He_NumScat%d_seed%d.h5'%(i, j),"r")
            weight = np.array(Data['weight'])
            proba += np.sum(weight)
        scatnum_proba[i-1] = proba
    return scatnum, scatnum_proba

# example
numNeutron = 3.4e+10
d2o_atomNum, v_atomNum = atomNum(d2o_r=1.0, d2o_h=14.3376, v_r=1.0, v_h=60.0)
scatnum, scatnum_proba = probaMultiScat('/data/yangni/cinemaRunData/2022_8_7/HW_D_1_NumBig/Z_NumScatSelect/HW_R1', 1, 85, 5)

plotStyle()
plt.semilogy(scatnum, np.power(10,np.log10(scatnum_proba/d2o_atomNum/numNeutron)), label=f'R=1.0mm, b=1, 3.4e+10 n')
plt.scatter(scatnum, scatnum_proba/d2o_atomNum/numNeutron, marker='o')
plt.xticks(scatnum)
plt.xlabel('Scattering Number')
plt.ylabel('Probability[/incident neutron/atom]')
plt.grid()
plt.legend(fontsize=12*1.5)
plt.tight_layout()
plt.show()