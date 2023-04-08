import numpy as np
from Cinema.Prompt import PromptFileReader
import matplotlib.pyplot as plt
from Cinema.Interface import plotStyle

# example for analyzing data in ScorerDeltaMomentum_SofQ_He_seed.mcpl.gz
file = PromptFileReader('/home/yangni/gitlabDev/cinema/gd2/ScorerDeltaMomentum_SofQ_He_seed4096.mcpl.gz')
hist_edge = file.getData('edge')
hist_content = file.getData('content')
hist_hit = file.getData('hit')
qtrue = hist_edge[:-1]
weight = hist_content/np.diff(hist_edge)
uncet = np.sqrt(hist_hit/10.)
err_weight = np.divide(weight, uncet, where=(uncet!=0.))

plotStyle()
plt.errorbar(qtrue, weight , yerr=err_weight, fmt='o-', label='D2O SQtrue')
plt.xlabel('Q')
plt.ylabel('S')
plt.title('DeltaMomentum_hist')
plt.grid()
plt.legend(fontsize=12*1.5)
plt.tight_layout()
plt.show()
