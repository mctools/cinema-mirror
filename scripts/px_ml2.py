#!/usr/bin/env python3

import numpy as np
from Cinema.PiXiu.io.cell import QeXmlCell
import phonopy
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)

# # cell = QeXmlCell('out_relax.xml')

# # q = np.array([[0,0,0],[0.1,0.,0],[0.,0.1,0],[0.,0.,0.1], [0.1,0.1,0.1]])

# # qabs = cell.qreduced2abs(q)
# # print(qabs)

# # qback = cell.qabs2reduced(qabs)
# # print(qback)


# s = 200
# # q = np.zeros( (s,3) )
# # a = np.linspace(0,0.5,s) 
# # q[:, 0] = a

# q = np.ones( (s,3) )
# a = np.linspace(0,0.5,s) 
# q= (q.T*a).T


# ph = phonopy.load(phonopy_yaml="phonopy.yaml", log_level=1, symmetrize_fc=True)
# ph.run_qpoints(q, with_eigenvectors=False)

# freq = ph.get_qpoints_dict()['frequencies'][:, :3]
# res = np.concatenate( (q,freq), axis=1)

# # ph.run_mesh(100., is_mesh_symmetry=False, with_eigenvectors=True)
# # mesh = ph.get_mesh_dict()
# # print('len', mesh.keys(), len(mesh['qpoints']))

    
# plt.plot(a, freq[:, 0], 'o')
# plt.plot(a, freq[:, 1], '+')
# plt.plot(a, freq[:, 2], 's')

# plt.show()


from Cinema.Interface.units import *
from Cinema.PiXiu.AtomInfo import getAtomMassBC

class CohPhon:
    def __init__(self, yamlfile = 'phonopy.yaml', cellQeRelaxXml='out_relax.xml', temperature=300.) -> None:
        self.ph = phonopy.load(phonopy_yaml=yamlfile, log_level=0, symmetrize_fc=True)
        self.cell = QeXmlCell(cellQeRelaxXml) # this should be changed to the experimental crystal size for production runs
        # self.pos=self.cell.lattice.dot(self.cell.reduced_pos.T).T
        self.pos=self.cell.reduced_pos.dot(self.cell.lattice)

        self.mass, self.bc, num = self.cell.getAtomInfo()
        self.sqMass = np.sqrt(self.mass)
        self.nAtom = len(self.sqMass)

        if any(self.ph.get_unitcell().get_atomic_numbers() != num):
            raise RuntimeError('Different unitcell in the DFT and phonon calculation ')

        self.temperature = temperature
        meshsize = 100.
        self.ph.run_mesh(meshsize, is_mesh_symmetry=False, with_eigenvectors=True)
        self.ph.run_thermal_displacement_matrices(self.temperature, self.temperature+1, 2, freq_min=0.002)
        # get_thermal_displacement_matrices returns the temperatures and thermal_displacement_matrices
        self.disp = self.ph.get_thermal_displacement_matrices()[1][0]
        omega = self.ph.get_mesh_dict()['frequencies']
        self.maxEn = omega.max()*THz*2*np.pi*hbar

        # print(self.disp)

    def calcMesh(self, meshsize):
        # # using iterator 
        # ph.init_mesh(10., is_mesh_symmetry=False, with_eigenvectors=True, use_iter_mesh=True)
        # for item in ph._mesh:
        #     q = ph._mesh._qpoints[ph._mesh._q_count-1]
        #     print('item',q, item)

        # keys are 'qpoints', 'weights', 'frequencies', 'eigenvectors'
        self.ph.run_mesh(meshsize, is_mesh_symmetry=False, with_eigenvectors=True)
        return self.ph.get_mesh_dict()
       
    def _calcPhonon(self, k):
        reducedK=self.cell.qabs2reduced(np.asarray(k))
        self.ph.run_qpoints(reducedK, with_eigenvectors=True)  

        res = self.ph.get_qpoints_phonon()
        # self.ph.write_hdf5_qpoints_phonon()
        nAtom = len(self.cell.element)

        en = res[0]*THz*2*np.pi*hbar
        eigv = res[1].reshape((-1, nAtom*3, nAtom, 3)) # en, eigenvector

        return en, eigv

    
    def _calcFormFact(self, Qarr, eigvecss, tau=None):
        # note, W = 0.5*Q*Q*u*u = 0.5*Q*Q*MSD


        # w =  -0.5 * Q.dot(np.swapaxes( Q.dot(self.disp), 1,2 ) )
        # print(w)

        F = np.zeros((Qarr.shape[0], self.nAtom*3))

        for i, (Q, eigvec) in enumerate(zip(Qarr, eigvecss)):
            w =  -0.5 * np.dot(np.dot(self.disp, Q), Q)

            # for testing
            # print('w', w, -0.5*self.disp[0,0,0]*np.linalg.norm(Q)**2)
            # print(self.disp[0])

            # print('Diff of using Q or tau', np.linalg.norm((self.bc/self.sqMass * np.exp(w + 1j*self.pos.dot(tau)) *eigvec.dot(Q)).sum(axis=1)) /
            #               np.linalg.norm((self.bc/self.sqMass * np.exp(w + 1j*self.pos.dot(Q)) *eigvec.dot(Q)).sum(axis=1)) )
            
            #summing for all atoms, F for each mode
            F[i]=np.abs((self.bc/self.sqMass * np.exp(w + 1j*self.pos.dot(Q)) *eigvec.dot(Q)).sum(axis=1))

        # print('eigvec', eigvec[1])
        return F    
    
    def s(self, Qin, reduced=False):
        if reduced:
            Q = self.cell.qreduced2abs(np.asarray(Qin))
        else:
            Q = np.asarray(Qin)

        if Q.ndim==1: 
            Q= np.expand_dims(Q, axis = 0)

        en, eigvec = self._calcPhonon(Q)

        if np.allclose(Q, np.zeros(3)):
            print('hit gamma')
            return None
        
        Qmag = np.linalg.norm(Q, axis=1) 

    
        F = self._calcFormFact(Q, eigvec)
    
        # not devided by the reciprocal volume so the unit is per atoms in the cell
        n = 1./(np.exp(en/(self.temperature*boltzmann))-1.)
        Smag = 0.5*(F*F)*hbar*hbar/en* (n + 1)

        # Smag = ((0.5*(F*F)*hbar*hbar).T/Qmag).T*0.0253
        
        if (en<=0).any():
            idx = en<=0
            Smag[idx] = 0.
            import warnings
            warnings.warn(f'en<=0, Q: {en<=0}, en: {en}', RuntimeWarning) 

        return Qmag, en, Smag


from time import time
import vegas

def toSpherical(x, y, z):
    r = np.sqrt(x*x+y*y+z*z)
    theta = np.arctan(y/x)
    phi = np.arccos(z/r)

    res = np.zeros([x.size,3])
    res[:, 0] = r
    res[:, 1] = theta
    res[:, 2] = phi
    return res

class kernel():
    def __init__(self, omegaBin=30) -> None:
        self.calc =  CohPhon()
        self.omegaRange = [0, self.calc.maxEn] 
        self.bin = omegaBin

    def setR(self, rmin, rmax):
        self.rmin = rmin
        self.rDelta = rmax-rmin
        print('rmin, rmax', rmin, rmax)

    def __call__(self, Q):
        
        Q, en, S = self.calc.s(Q)
        if (S<0.).any():
            print('S<0.', Q, S)
            raise RuntimeError()
        
        I = S.sum()
        
        dI, bin_edges = np.histogram(en, bins=self.bin, range=self.omegaRange, weights=(S.T).T)
        return dict(I=I, dI=dI)

        


qSize = 1
maxQ = 0.01
enSize = 80
qEdge=np.linspace(0.0, maxQ, qSize+1)
Q = qEdge[:-1]+np.diff(qEdge)*0.5

# per reciprocal space
q_volume = qEdge**3*np.pi*4/3. # spher volumes
q_volume_diff = np.diff(q_volume)


k = kernel(enSize)
enEdge = np.linspace(0, k.calc.maxEn, enSize+1 )
en = enEdge[:-1]+np.diff(enEdge)*0.5
# per energy
en_diff = np.diff(en).mean()


sqw = np.zeros([qSize, enSize])

#map = vegas.AdaptiveMap([[0, 1], [0, 2*np.pi], [0, np.pi]])     # uniform map
#x = np.random.normal(loc=0, scale=(qEdge[1]-qEdge[0])/5, size=(10000, 3))
#sph = toSpherical(x[:,0], x[:,1], x[:,2])
#map.adapt_to_samples(sph, k(sph), nitn=5)       # precondition map
#integ = vegas.Integrator(map, alpha=0.5, nproc = 10)

for i in range(qSize):
    t1 = time()
    # k.setR(qEdge[i], qEdge[i+1])   

    integ =  vegas.Integrator([[qEdge[i], qEdge[i+1]], [0, np.pi], [0, np.pi]], nproc = 4)

    integ(k, nitn=10, neval=5000)
    result = integ(k, nitn=10, neval=10000, adapt = False)
    for j in range(enSize):
        sqw[i,j] = (result['dI'][j] / result['I']).mean

    sqw[i] *= 1./(q_volume_diff[i]*en_diff)


    print(result.summary())
    print('   Q =', Q[i])
    print('   I =', result['I'])
    print('sqw[i] =', sqw[i])
    print('dI/I =', result['dI'] / result['I'])
    print('sum(dI/I) =', sum(result['dI']) / result['I'])
    t2 = time()
    print(f'Function  executed in {(time()-t1):.4f}s\n')




import matplotlib.pyplot as plt
import matplotlib.colors as colors
from Cinema.Interface import plotStyle
plotStyle()
fig=plt.figure()
ax = fig.add_subplot(111)
H = sqw.T

X, Y = np.meshgrid(Q, en)
pcm = ax.pcolormesh(X, Y, H, cmap=plt.cm.jet, shading='auto')
fig.colorbar(pcm, ax=ax)
plt.grid()
plt.savefig(fname='lin.pdf')

fig=plt.figure()
ax = fig.add_subplot(111)
pcm = ax.pcolormesh(X, Y, H, cmap=plt.cm.jet, norm=colors.LogNorm(vmin=H.max()*1e-10, vmax=H.max()), shading='auto')
plt.grid()
plt.savefig(fname='log.pdf')

plt.show()


