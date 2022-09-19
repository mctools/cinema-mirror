import numpy as np
from yn_ana_code import SQTrueHist, MultiScatAnalysor, atomNum, SQHist
from Cinema.Interface import plotStyle
import matplotlib.pyplot as plt

# example for analyzing sqtrue of d2o in 1Dhist    
numNeutron = 3.4e+10
d2o_atomNum, v_atomNum = atomNum(d2o_r=1.0, d2o_h=14.3376, v_r=1.0, v_h=60.0)
d2o = SQTrueHist(xmin=0.1, xmax=120.0, numbin=150, linear=False)
d2o_qtrue, d2o_s = d2o.fillHist("/data/yangni/cinemaRunData/2022_8_7/HW_D_1_NumBig/HW_R1", 1, 85)
v = SQTrueHist(xmin=0.1, xmax=120.0, numbin=150, linear=False)
v_qtrue, v_s = v.fillHist("/data/yangni/cinemaRunData/2022_8_7/HW_D_1_NumBig/V_R1", 1, 85)
d2o_s_calib = np.divide(d2o_s/d2o_atomNum/numNeutron, v_s/v_atomNum/numNeutron, where=v_s!=0.)

# example for decomposing sqtrue of d2o in 1Dhist  
d2oMS = MultiScatAnalysor("/data/yangni/cinemaRunData/2022_8_7/HW_D_1_NumBig/HW_R1", 1, 85)
d2oMS1_qtrue, d2oMS1_s = d2oMS.sqHist(xmin=0.1, xmax=120.0, numbin=150, scatnum=1, inelastic=True, linear=False)
d2oMS1_s_calib = np.divide(d2oMS1_s/d2o_atomNum/numNeutron, v_s/v_atomNum/numNeutron, where=v_s!=0.)
d2oMS2_qtrue, d2oMS2_s = d2oMS.sqHist(xmin=0.1, xmax=120.0, numbin=150, scatnum=2, inelastic=True, linear=False)
d2oMS2_s_calib = np.divide(d2oMS2_s/d2o_atomNum/numNeutron, v_s/v_atomNum/numNeutron, where=v_s!=0.)
d2oMS3_qtrue, d2oMS3_s = d2oMS.sqHist(xmin=0.1, xmax=120.0, numbin=150, scatnum=3, inelastic=True, linear=False)
d2oMS3_s_calib = np.divide(d2oMS3_s/d2o_atomNum/numNeutron, v_s/v_atomNum/numNeutron, where=v_s!=0.)
probability = d2oMS.probaMultiScat(scatnum=2) # calculate the probability of  scattering number = 2
print(probability)


plotStyle()
plt.xlim(0.0, 16.0)
plt.plot(d2o_qtrue, d2o_s_calib, label=f'R_D$_2$O=1mm, b=1, Total ')
plt.plot(d2oMS1_qtrue, d2oMS1_s_calib, label=f'R_D$_2$O=1mm, b=1, 1Scattering qtrue')
plt.plot(d2oMS2_qtrue, d2oMS2_s_calib, label=f'R_D$_2$O=1mm, b=1, 2Scatterings')
plt.plot(d2oMS3_qtrue, d2oMS3_s_calib, label=f'R_D$_2$O=1mm, b=1, 3Scatterings')
plt.xlabel('Q[Å$^-$$^1$]')
plt.ylabel('S[/incident particle/atom]')
plt.grid()
plt.legend(fontsize=12*1.5)
plt.tight_layout()
plt.show()
