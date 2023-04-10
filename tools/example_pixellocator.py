import numpy as np
from Cinema.Experiment.Analyser import PixelLocator, IDFLoader


# file = '/home/yangni/Work/MPIGDML/Data/idf/idf/module10702.txt'
# pxlID = np.loadtxt(file, dtype=int, usecols=(0))
# pxlPos = np.loadtxt(file, usecols=(1,2,3))*1e3 #meter to mm
# particlePos = np.array([-0.505989, -0.199400, -1.226291])*1e3 #meter to mm
# pixel = PixelLocator(pxlID, pxlPos)
# id, dist = pixel.locate(particlePos)
# print(id, dist)

idf = IDFLoader('/home/yangni/Work/MPIGDML/Data/idf/idf')
tree = idf.query('module10701')
dd = tree.generateDD('/data/yangni/cinemaRunData/2022_6_02/HW_Num_4_10_10/*300*.wgt')
print(dd.dd[257].keys())