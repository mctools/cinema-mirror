import numpy as np
import matplotlib.pyplot as plt
x=np.load('ScorerNeutronSq_SofQ_seed4096_edge.npy')
y=np.load('ScorerNeutronSq_SofQ_seed4096_content.npy')
plt.loglog(x[:-1],y/np.diff(x), label=f'integral={y.sum()}')
plt.grid()
plt.legend()
plt.show()
