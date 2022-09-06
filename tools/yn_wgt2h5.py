import h5py
import numpy as np
from  Cinema.Prompt.Math.Hist import Hist1D
import matplotlib.pyplot as plt

# store the data in h5py file
def wgt2h5(filePath, seedStart, seedEnd):
    for i in range(seedStart, seedEnd):
        Data=np.loadtxt(filePath+'/ScororNeutronSq_SofQ_He_seed%d.wgt'%i)
        h5file=h5py.File("ScororNeutronSq_SofQ_He_seed%d.h5"%i,"w")
        h5file.create_dataset('tof_us',data=Data[:,0])
        h5file.create_dataset('position_x',data=Data[:,1])
        h5file.create_dataset('position_y',data=Data[:,2])
        h5file.create_dataset('position_z',data=Data[:,3])
        h5file.create_dataset('q',data=Data[:,4]) 
        h5file.create_dataset('qtrue',data=Data[:,5])
        h5file.create_dataset('ekin',data=Data[:,6]) 
        h5file.create_dataset('EKin0',data=Data[:,7]) 
        h5file.create_dataset('EKin',data=Data[:,8])
        h5file.create_dataset('numScat',data=Data[:,9])
        h5file.create_dataset('weight',data=Data[:,10])
        h5file.close()  

