#!/usr/bin/env python3

import numpy as np
np.set_printoptions(suppress=True)
from Cinema.PiXiu.io.cell import QeXmlCell
import phonopy
import matplotlib.pyplot as plt
import pickle
import argparse
from Cinema.Interface.units import *
from Cinema.PiXiu.AtomInfo import getAtomMassBC


from time import time
import vegas

def dumpObj2File(fn, obj):
    f = open (fn, 'wb')
    pickle.dump(obj, f)
    f.close()



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
        self.disp = np.copy(self.ph.get_thermal_displacement_matrices()[1][0])
        omega = np.copy(self.ph.get_mesh_dict()['frequencies'])
        self.maxHistEn = omega.max()*THz*2*np.pi*hbar + 0.005 #add 5meV as the energy margin
        del self.ph
        self.ph = phonopy.load(phonopy_yaml=yamlfile, log_level=0, symmetrize_fc=True)

        # print(self.disp)

    def getGammaPoint(self, Qin):
        Q = np.atleast_2d(Qin)
        # print('Q', Q)
        Qred = self.cell.qabs2reduced(Q)
        # print('rudu', Qred)

        ceil = np.ceil(Qred)
        floor = np.floor(Qred)

        gamma = np.zeros((len(Q), 3))

        for i, (low, up) in enumerate(zip(floor, ceil)):
            gpoints_reduced = np.array([[low[0], low[1], low[2]],
                                [low[0], low[1], up[2]],
                                [low[0], up[1], low[2]],
                                [low[0], up[1], up[2]],
                                [up[0], low[1], low[2]],
                                [up[0], low[1], up[2]],
                                [up[0], up[1], low[2]],
                                [up[0], up[1], up[2]]])
            gpoints = self.cell.qreduced2abs(gpoints_reduced)
            dist = gpoints-Q[i]
            dist = (dist*dist).sum(axis=1)
            # print('dist', dist, np.argmin(dist) )
            gamma[i] = gpoints[np.argmin(dist)]

        return gamma

        # print('gamma', gamma, self.cell.qabs2reduced(gamma))
        # print('Qback', self.cell.qreduced2abs(Qred))

        # # raise RuntimeError()
        # if len(Q) > 1:
        #     raise RuntimeError()


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

        gamma = self.getGammaPoint(Qarr)

        for i, (g, Q, eigvec) in enumerate(zip(gamma, Qarr, eigvecss)):
            w =  -0.5 * np.dot(np.dot(self.disp, Q), Q)

            # for testing
            # print('w', w, -0.5*self.disp[0,0,0]*np.linalg.norm(Q)**2)
            # print(self.disp[0])

            # print('Diff of using Q or tau', np.linalg.norm((self.bc/self.sqMass * np.exp(w + 1j*self.pos.dot(tau)) *eigvec.dot(Q)).sum(axis=1)) /
            #               np.linalg.norm((self.bc/self.sqMass * np.exp(w + 1j*self.pos.dot(Q)) *eigvec.dot(Q)).sum(axis=1)) )
            
            #summing for all atoms, F for each mode
            F[i]=np.abs((self.bc/self.sqMass * np.exp(w + 1j*self.pos.dot(g)) *eigvec.dot(Q)).sum(axis=1))

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

        tinyphon = en < 1e-5 # in eV, cufoof small phonons
        en[tinyphon] = 1. # replease small phonon energy to 1eV to avoid diveded by zero RuntimeWarning
        
        Qmag = np.linalg.norm(Q, axis=1)  
        F = self._calcFormFact(Q, eigvec)

        # the unit is per atoms by not per reciprocal volume
        n = 1./(np.exp(en/(self.temperature*boltzmann))-1.)

        Smag = 0.5*(F*F)*hbar*hbar/en* (n + 1)
        if tinyphon.any(): # fill zero for small phonons
            Smag[tinyphon] = 0.   

        self.ph.set_partial_DOS                

        return Qmag, en, Smag

class kernel(vegas.BatchIntegrand):
    def __init__(self, omegaBin=30, temp=300.) -> None:
        self.calc =  CohPhon(temperature=temp)
        self.omegaRange = [0, self.calc.maxHistEn] 
        self.bin = omegaBin

    # def setR(self, rmin, rmax):
    #     self.rmin = rmin
    #     self.rDelta = rmax-rmin
    #     print('rmin, rmax', rmin, rmax)

    def __call__(self, input):
        # x: rho, theta(0, 2pi), phi (0, pi)
        x=np.atleast_2d(input)
        r=x[:,0]#*self.rDelta + self.rmin
        theta=x[:,1]
        phi=x[:,2]
        
        
        # https://mathworld.wolfram.com/SphericalCoordinates.html
        sin_theta=np.sin(theta)
        cos_theta=np.cos(theta)
        sin_phi=np.sin(phi)
        cos_phi=np.cos(phi)

        pos = np.zeros_like(x)
        pos[:, 0] = cos_theta*sin_phi
        pos[:, 1] = sin_theta*sin_phi
        pos[:, 2] = cos_phi
        pos = (pos.T*r).T
        
        Q, en, S = self.calc.s(pos)
        if (S<0.).any():
            print('S<0.', pos, S)
            raise RuntimeError()
        

        factor = r*r*sin_phi
        qvec2q = 1./(4.*np.pi)

        contr = (S.T*factor).T*qvec2q

        I = contr.sum(axis=1)
        
        # return I
        
        dI = np.zeros((I.size, self.bin))
        for i in range(I.size):
            dI[i], _ = np.histogram(en[i], bins=self.bin, range=self.omegaRange, weights=contr[i])

        return dict(I=I, dI=dI)


def gen_parser():
    #parameters
    #temperature in kelvin
    #upper limit for the Q, maxQ, float
    #frequency bin size for the histogram, freSize, int
    #Q bin size for the histogram, QSize, int
    #stepping for the hkl, int
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--temperature', action='store', type=float,
                        dest='temp', help='temperature in kelvin', required=True)
    parser.add_argument('-u', '--upper-limit-Q', action='store', type=float, default=10.0,
                        dest='maxQ', help='upper limit for the Q')
    parser.add_argument('-q', '--Q-bin-size', action='store', type=int, default=300,
                        dest='QSize', help='Q bin size for the histogram')
    parser.add_argument('-e', '--en-bin-size', action='store', type=int, default=200,
                        dest='enSize', help='energy bin size for the histogram')
    parser.add_argument('-o', '--output-file-name', action='store', default='qehist.h5',
                        dest='output', help='output file name')
    parser.add_argument('-n', '--neval', action='store', type=float, default=10000,
                    dest='neval', help='number of evaluation for the 20 iterations')
    parser.add_argument('-p', '--partitions', action='store', type=int, default=1,
                    dest='partitions', help='number of partitions.')
    return parser
    
args=gen_parser().parse_args()
temp = args.temp #temperature in kelvin
maxQ = args.maxQ
enSize = args.enSize
qSize = args.QSize
output = args.output
partitions = args.partitions
neval=int(args.neval)
   

qEdge=np.linspace(0.0, maxQ, qSize+1)
Q = qEdge[:-1]+np.diff(qEdge)*0.5

# per reciprocal space
q_volume = qEdge**3*np.pi*4/3. # spher volumes
q_volume_diff = np.diff(q_volume)


k = kernel(enSize, temp)
enEdge = np.linspace(0, k.calc.maxHistEn, enSize+1 )
en = enEdge[:-1]+np.diff(enEdge)*0.5
# per energy
en_diff = np.diff(en).mean()
sqw = np.zeros([qSize, enSize])


def run(i):
    print('running ', i)
    t1 = time()
    sqw = np.zeros(enSize)

    # k.setR(qEdge[i], qEdge[i+1])   

    integ =  vegas.Integrator([[qEdge[i], qEdge[i+1]], [0, np.pi], [0, np.pi]])

    integ(k, nitn=10, neval=neval)
    result = integ(k, nitn=10, neval=neval, adapt = False)

    dumpObj2File(f'result_{i}.pkl', result)

    for j in range(enSize):
        sqw[j] = (result['dI'][j]).mean

    sqw *= 1./(q_volume_diff[i]*en_diff)
    sqw *= 2. # theata is only for half of the sphere

    # print(result.summary())
    # print('   Q =', Q[i])
    # print('   I =', result['I'])
    # print('sqw =', sqw)
    # print('sum(dI/I) =', sum(result['dI']) / result['I'])
    # print('density /Q/energy', result['I']/(q_volume_diff[i]*en_diff))
    # print('nstrat shape', np.array(integ.nstrat) )
    t2 = time()
    print(f'Run {i}: Q={qEdge[i]}, {qEdge[i+1]}: {np.mean([qEdge[i], qEdge[i+1]])}  executed in {(time()-t1):.2f}s, I=', result['I'], f', chi2={result.chi2/result.dof:.2f}, Q={result.Q} \n')
    return sqw


from multiprocessing import Pool
pool = Pool(partitions)
start_time = time()

with pool:
    res = pool.map(run, range(qSize))
    for i, p in enumerate(res):
        sqw[i] = p


finish_time = time()
print(f"Program finished in {finish_time-start_time} seconds")



# save data 
import h5py
f0=h5py.File(output,"w")

## coherent 
f0.create_dataset("q", data=Q, compression="gzip")
f0.create_dataset("en", data=en, compression="gzip")
f0.create_dataset("s", data=sqw, compression="gzip")

f0.close()


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

# plt.show()
