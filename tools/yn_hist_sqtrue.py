import numpy as np
from yn_ana_code import SQAnaHist1D, atomNum
from Cinema.Interface import plotStyle
import matplotlib.pyplot as plt

# example for analyzing sqtrue of d2o in 1Dhist
numNeutron = 3.4e+10
d2o_atomNum, v_atomNum = atomNum(d2o_r=1.0, d2o_h=14.3376, v_r=1.0, v_h=60.0)
d2o = SQAnaHist1D("/data/yangni/cinemaRunData/2022_8_7/HW_D_1_NumBig/HW_R1", 1, 85)
v = SQAnaHist1D("/data/yangni/cinemaRunData/2022_8_7/HW_D_1_NumBig/V_R1", seedStart=1, seedEnd=85)
d2o_qtrue, d2o_s, d2ohist = d2o.sqtrueHist(xmin=0.1, xmax=120.0, numbin=150, linear=False)
v_qtrue, v_s, vhist = v.sqtrueHist(xmin=0.1, xmax=120.0, numbin=150, linear=False)
d2o_s_calib = np.divide(d2o_s/d2o_atomNum/numNeutron, v_s/v_atomNum/numNeutron, where=v_s!=0.)

# example for analyzing sq of d2o in 1Dhist
# d2o_qtrue, d2o_s, d2ohist = d2o.sqHist(xmin=0.1, xmax=120.0, numbin=1500, linear=True)
# v_qtrue, v_s, vhist = v.sqHist(xmin=0.1, xmax=120.0, numbin=1500, linear=True)
# d2o_s_calib = np.divide(d2o_s, v_s, where=v_s!=0.)

plotStyle()
plt.xlim(0.0, 16.0)
plt.plot(d2o_qtrue, d2o_s_calib, label=f'Simulation, R_D$_2$O=1mm, b=1')
plt.xlabel('Q[Å$^-$$^1$]')
plt.ylabel('S[/incident particle/atom]')
plt.grid()
plt.legend(fontsize=12*1.5)
plt.tight_layout()
plt.show()

