import numpy as np
from  Cinema.Prompt.Math.Hist import Hist1D
import h5py

# sq for elastic scattering
def sqHist(filePath, seedStart, seedEnd, xmin, xmax, numbin, linear=True):
    sq_hist = Hist1D(xmin, xmax, numbin, linear)
    for i in range(seedStart, seedEnd+1):
        # Data = np.loadtxt(filePath+'/ScororNeutronSq_SofQ_He_seed%d.wgt'%i)
        # qtrue = np.ascontiguousarray(Data[:,5])
        # weight = np.ascontiguousarray(Data[:,10])
        Data = h5py.File(filePath+"/ScororNeutronSq_SofQ_He_seed%d.h5"%i,"r")
        # tof_us = np.array(Data['tof_us'])
        # position_x = np.array(Data['position_x'])
        # position_y = np.array(Data['position_y'])
        # position_z = np.array(Data['position_z'])
        q = np.array(Data['q'])
        # qtrue = np.array(Data['qtrue'])
        # ekin = np.array(Data['ekin'])
        # EKin0 = np.array(Data['EKin0'])
        # EKin = np.array(Data['EKin'])
        # numScat = np.array(Data['numScat'])
        weight = np.array(Data['weight'])
        sq_hist.fillmany(q, weight)
        
    q_hist = sq_hist.getCentre()
    s_hist = sq_hist.getWeight()/np.diff(sq_hist.getEdge())  
    return q_hist, s_hist, sq_hist 

# sqtrue for inelastic scattering
def sqtrueHist(filePath, seedStart, seedEnd, xmin, xmax, numbin, linear=True):
    sqtrue_hist = Hist1D(xmin, xmax, numbin, linear)
    for i in range(seedStart, seedEnd+1):
        # Data = np.loadtxt(filePath+'/ScororNeutronSq_SofQ_He_seed%d.wgt'%i)
        # qtrue = np.ascontiguousarray(Data[:,5])
        # weight = np.ascontiguousarray(Data[:,10])
        Data = h5py.File(filePath+"/ScororNeutronSq_SofQ_He_seed%d.h5"%i,"r")
        # tof_us = np.array(Data['tof_us'])
        # position_x = np.array(Data['position_x'])
        # position_y = np.array(Data['position_y'])
        # position_z = np.array(Data['position_z'])
        # q = np.array(Data['q'])
        qtrue = np.array(Data['qtrue'])
        # ekin = np.array(Data['ekin'])
        # EKin0 = np.array(Data['EKin0'])
        # EKin = np.array(Data['EKin'])
        # numScat = np.array(Data['numScat'])
        weight = np.array(Data['weight'])
        sqtrue_hist.fillmany(qtrue, weight)   
        
    qtrue_hist = sqtrue_hist.getCentre()
    s_hist = sqtrue_hist.getWeight()/np.diff(sqtrue_hist.getEdge())  
    return qtrue_hist, s_hist, sqtrue_hist 

# decomposition of sqtrue
def sqtrueMultiScatHist(filePath, seedStart, seedEnd, scatnum_max, xmin, xmax, numbin, linear=True):
    s_multiscat = np.empty(shape=[0, numbin], dtype=float)
    for i in range(1, scatnum_max+1):
        sqtrue_hist = Hist1D(xmin, xmax, numbin, linear)
        for j in range(seedStart, seedEnd+1):
            Data = h5py.File(filePath+'/ScororNeutronSq_SofQ_He_NumScat%d_seed%d.h5'%(i, j),"r")
            qtrue = np.array(Data['qtrue'])
            weight = np.array(Data['weight'])
            sqtrue_hist.fillmany(qtrue, weight)
        s_hist = sqtrue_hist.getWeight()/np.diff(sqtrue_hist.getEdge()) 
        s_multiscat = np.append(s_multiscat, [s_hist], axis=0)
    qtrue_multiscat = sqtrue_hist.getCentre()
    return qtrue_multiscat, s_multiscat


# atom number in material        
def atomNum(d2o_r, d2o_h, v_r, v_h): # mm
    avogadro = 6.02e+23
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