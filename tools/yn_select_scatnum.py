import h5py
import numpy as np

def scatnumSelect(filePath, seedStart, seedEnd):
    for i in range(seedStart, seedEnd+1):
        print(i)
        HW_ExpR=h5py.File(filePath+"/ScororNeutronSq_SofQ_He_seed%d.h5"%i,"r")
        q=np.array(HW_ExpR['q'])
        qtrue=np.array(HW_ExpR['qtrue'])
        numScat=np.array(HW_ExpR['numScat'])
        weight=np.array(HW_ExpR['weight'])
        HW_ExpR.close()
        num=numScat.shape[0] # the number of particles which be recorded in the current h5 file
        index_NumScat1=np.empty([0,],int)
        index_NumScat2=np.empty([0,],int)
        index_NumScat3=np.empty([0,],int)
        index_NumScat4=np.empty([0,],int)
        index_NumScat5=np.empty([0,],int)
        index_NumScat6=np.empty([0,],int)
        index_NumScat7=np.empty([0,],int)
        index_NumScat8=np.empty([0,],int)
        index_NumScat9=np.empty([0,],int)
        index_NumScat10=np.empty([0,],int)

        for j in range(0, num):
            if numScat[j]==1:
                    index_NumScat1=np.append(index_NumScat1,j)
            elif numScat[j]==2:
                    index_NumScat2=np.append(index_NumScat2,j)
            elif numScat[j]==3:
                    index_NumScat3=np.append(index_NumScat3,j)
            elif numScat[j]==4:
                    index_NumScat4=np.append(index_NumScat4,j)
            elif numScat[j]==5:
                    index_NumScat5=np.append(index_NumScat5,j)
            elif numScat[j]==6:
                    index_NumScat6=np.append(index_NumScat6,j)
            elif numScat[j]==7:
                    index_NumScat7=np.append(index_NumScat7,j)
            elif numScat[j]==8:
                    index_NumScat8=np.append(index_NumScat8,j)
            elif numScat[j]==9:
                    index_NumScat9=np.append(index_NumScat9,j)
            elif numScat[j]==10:
                    index_NumScat10=np.append(index_NumScat10,j)

        NumScat1=h5py.File("ScororNeutronSq_SofQ_He_NumScat1_seed%d.h5"%i,"w")
        NumScat1.create_dataset('q',data=q[index_NumScat1])
        NumScat1.create_dataset('qtrue',data=qtrue[index_NumScat1])
        NumScat1.create_dataset('numScat',data=numScat[index_NumScat1])
        NumScat1.create_dataset('weight',data=weight[index_NumScat1])
        NumScat1.close() 
        NumScat2=h5py.File("ScororNeutronSq_SofQ_He_NumScat2_seed%d.h5"%i,"w")
        NumScat1.create_dataset('q',data=q[index_NumScat2])
        NumScat2.create_dataset('qtrue',data=qtrue[index_NumScat2])
        NumScat2.create_dataset('numScat',data=numScat[index_NumScat2])
        NumScat2.create_dataset('weight',data=weight[index_NumScat2])
        NumScat2.close()
        NumScat3=h5py.File("ScororNeutronSq_SofQ_He_NumScat3_seed%d.h5"%i,"w")
        NumScat1.create_dataset('q',data=q[index_NumScat3])
        NumScat3.create_dataset('qtrue',data=qtrue[index_NumScat3])
        NumScat3.create_dataset('numScat',data=numScat[index_NumScat3])
        NumScat3.create_dataset('weight',data=weight[index_NumScat3])
        NumScat3.close()
        NumScat4=h5py.File("ScororNeutronSq_SofQ_He_NumScat4_seed%d.h5"%i,"w")
        NumScat1.create_dataset('q',data=q[index_NumScat4])
        NumScat4.create_dataset('qtrue',data=qtrue[index_NumScat4])
        NumScat4.create_dataset('numScat',data=numScat[index_NumScat4])
        NumScat4.create_dataset('weight',data=weight[index_NumScat4])
        NumScat4.close()
        NumScat5=h5py.File("ScororNeutronSq_SofQ_He_NumScat5_seed%d.h5"%i,"w")
        NumScat1.create_dataset('q',data=q[index_NumScat5])
        NumScat5.create_dataset('qtrue',data=qtrue[index_NumScat5])
        NumScat5.create_dataset('numScat',data=numScat[index_NumScat5])
        NumScat5.create_dataset('weight',data=weight[index_NumScat5])
        NumScat5.close()
        NumScat6=h5py.File("ScororNeutronSq_SofQ_He_NumScat6_seed%d.h5"%i,"w")
        NumScat1.create_dataset('q',data=q[index_NumScat6])
        NumScat6.create_dataset('qtrue',data=qtrue[index_NumScat6])
        NumScat6.create_dataset('numScat',data=numScat[index_NumScat6])
        NumScat6.create_dataset('weight',data=weight[index_NumScat6])
        NumScat6.close()
        NumScat7=h5py.File("ScororNeutronSq_SofQ_He_NumScat7_seed%d.h5"%i,"w")
        NumScat1.create_dataset('q',data=q[index_NumScat7])
        NumScat7.create_dataset('qtrue',data=qtrue[index_NumScat7])
        NumScat7.create_dataset('numScat',data=numScat[index_NumScat7])
        NumScat7.create_dataset('weight',data=weight[index_NumScat7])
        NumScat7.close()
        NumScat8=h5py.File("ScororNeutronSq_SofQ_He_NumScat8_seed%d.h5"%i,"w")
        NumScat1.create_dataset('q',data=q[index_NumScat8])
        NumScat8.create_dataset('qtrue',data=qtrue[index_NumScat8])
        NumScat8.create_dataset('numScat',data=numScat[index_NumScat8])
        NumScat8.create_dataset('weight',data=weight[index_NumScat8])
        NumScat8.close()
        NumScat9=h5py.File("ScororNeutronSq_SofQ_He_NumScat9_seed%d.h5"%i,"w")
        NumScat1.create_dataset('q',data=q[index_NumScat9])
        NumScat9.create_dataset('qtrue',data=qtrue[index_NumScat9])
        NumScat9.create_dataset('numScat',data=numScat[index_NumScat9])
        NumScat9.create_dataset('weight',data=weight[index_NumScat9])
        NumScat9.close()
        NumScat10=h5py.File("ScororNeutronSq_SofQ_He_NumScat10_seed%d.h5"%i,"w")
        NumScat1.create_dataset('q',data=q[index_NumScat10])
        NumScat10.create_dataset('qtrue',data=qtrue[index_NumScat10])
        NumScat10.create_dataset('numScat',data=numScat[index_NumScat10])
        NumScat10.create_dataset('weight',data=weight[index_NumScat10])
        NumScat10.close()
 
 # example       
scatnumSelect("/data/yangni/cinemaRunData/2022_8_7/HW_D_1_NumBig/HW_R1", seedStart=1, seedEnd=8)