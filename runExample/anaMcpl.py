import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from Cinema.Interface import plotStyle
from Cinema.Prompt.PromptFileReader import McplAnalysor

File = './*ScorerPSD_Monitor1_seed*.mcpl.gz'   
Data = McplAnalysor(File)
hist = Data.getHistMany(offset=0, num=None)

plotStyle()
# Fig lamda_theta_hw_Exp
fig = plt.figure()
ax = fig.add_subplot(111)
H = hist.weight.T
x = hist.xcentre
y = hist.ycentre
X, Y = np.meshgrid(x, y)
pcm = ax.pcolormesh(X, Y, H, cmap=plt.cm.jet, norm=colors.LogNorm(vmin=H.max()*1e-2, vmax=H.max()), shading='auto')
fig.colorbar(pcm, ax=ax)
plt.grid()
plt.title(f'integral {H.sum()}')
plt.tight_layout()
plt.show()


