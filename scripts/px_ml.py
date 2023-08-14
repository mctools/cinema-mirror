#!/usr/bin/env python3

import numpy as np
from Cinema.PiXiu.io.cell import QeXmlCell
import phonopy
import matplotlib.pyplot as plt
import pickle

def dumpObj2File(fn, obj):
    f = open (fn, 'wb')
    pickle.dump(obj, f)
    f.close()

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
            print(Q)
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

class kernel(vegas.BatchIntegrand):
    def __init__(self, omegaBin=30) -> None:
        self.calc =  CohPhon()
        self.omegaRange = [0, self.calc.maxEn] 
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

        contr = (S.T*factor).T

        I = contr.sum(axis=1)
        
        # return I
        
        dI = np.zeros((I.size, self.bin))
        for i in range(I.size):
            dI[i], _ = np.histogram(en[i], bins=self.bin, range=self.omegaRange, weights=contr[i])

        return dict(I=I, dI=dI)

        


qSize = 80
maxQ = 30
enSize = 200
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

import multiprocessing

def run(i):
    print('running ', i)
    t1 = time()
    sqw = np.zeros(enSize)

    # k.setR(qEdge[i], qEdge[i+1])   

    integ =  vegas.Integrator([[qEdge[i], qEdge[i+1]], [0, np.pi], [0, np.pi]])

    integ(k, nitn=10, neval=500)
    result = integ(k, nitn=10, neval=1000, adapt = False)

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
    print(f'Run{i} Q={np.mean([qEdge[i], qEdge[i+1]])}  executed in {(time()-t1):.2f}s, I=', result['I'], f', chi2={result.chi2/result.dof:.2f}, Q={result.Q} \n')
    return sqw


from multiprocessing import Pool
p = Pool(1)
start_time = time()

with p:
    res = p.map(run, range(qSize))
    for i, p in enumerate(res):
        sqw[i] = p


# pool = multiprocessing.Pool(8)
# processes = [pool.apply_async(run, args=(i,)) for i in range(qSize)]
# # result = [p.get() for p in processes]
# for i, p in enumerate(processes):
#     sqw[i] = p.get()

finish_time = time()
print(f"Program finished in {finish_time-start_time} seconds")


# for i in range(qSize):
#     t1 = time()
#     # k.setR(qEdge[i], qEdge[i+1])   

#     integ =  vegas.Integrator([[qEdge[i], qEdge[i+1]], [0, np.pi], [0, np.pi]])

#     integ(k, nitn=10, neval=500)
#     result = integ(k, nitn=10, neval=1000, adapt = False)
#     for j in range(enSize):
#         sqw[i,j] = (result['dI'][j]).mean

#     sqw[i] *= 1./(q_volume_diff[i]*en_diff)
#     sqw[i] *= 2. # theata is only for half of the sphere


#     print(result.summary())
#     print('   Q =', Q[i])
#     print('   I =', result['I'])
#     print('sqw[i] =', sqw[i])
#     print('sum(dI/I) =', sum(result['dI']) / result['I'])
#     print('density /Q/energy', result['I']/(q_volume_diff[i]*en_diff))
#     t2 = time()
#     print('nstrat shape', np.array(integ.nstrat) )
#     print(f'Function  executed in {(time()-t1):.4f}s\n')

dumpObj2File(f'Sqw.pkl', sqw)
dumpObj2File(f'Q.pkl', Q)
dumpObj2File(f'en.pkl', en)

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




### debug
# Q, en, S = calc.s([-0.03460482,  0.03460482,  0.03460482], reduced=False)
# print('s', S[0] )

# Q, en, S = calc.s([[-0.03460482,  0.03460482,  0.03460482], [-0.03460482,  0.03460482,  0.03460482]], reduced=False)
# print('s', S[0] )

# calc = CohPhon()

# t1 = time()
# Q, en, S = calc.s( np.random.random((100000,3)), reduced=False)
# t2 = time()
# print(f'Function  executed in {(time()-t1):.4f}s')


# ## Lebedev rule
# from Cinema.Interface.Utils import findData

# def getQ(Qmag):
#     lev = findData('lebedev/lebedev_041.txt')
#     data = np.loadtxt(lev)
#     deg = np.pi/180.
#     sin_theta = np.sin(data[::2,0]*deg)
#     cos_theta = np.cos(data[::2,0]*deg)
#     sin_phi = np.sin(data[::2,1]*deg)
#     cos_phi = np.cos(data[::2,1]*deg)
#     weight = data[::2, 2]*2

#     # https://people.sc.fsu.edu/~jburkardt/cpp_src/sphere_lebedev_rule/sphere_lebedev_rule.html
#     # integral f(x,y,z) = 4 * pi * sum ( 1 <= i <= N ) f(xi,yi,zi) * w(xi,yi,zi)

#     cor = np.zeros((weight.size, 3))
#     cor[:,0] = cos_theta*sin_phi
#     cor[:,1] = sin_theta* sin_phi
#     cor[:,2] = cos_phi
#     cor[np.abs(cor)<1e-14] = 0. # truncate small numbers to zero
#     return cor*Qmag, weight

# numPoints = len(getQ(1)[0])

# batchSize = 500
# calc = CohPhon()
# Q=np.zeros(numPoints)
# en=np.zeros((batchSize, numPoints, calc.nAtom*3))
# S=np.zeros_like(en)

# print(print(calc._calcPhonon([0,0,0])))


# # SQ
# t1 = time()
# Qvec = np.linspace(0.1, 30.1, batchSize)
# for i, Qmag in enumerate(Qvec):
#     qv, w = getQ(Qmag)
#     Q, en[i], S[i] = calc.s(qv)
#     surfArea = Qmag*Qmag # 4pi is reduced
#     S[i] = (S[i].T*w).T/surfArea
    
# print(f'Function  executed in {(time()-t1):.4f}s')

# plt.plot(Qvec, S.sum(axis=1).sum(axis=1))
# plt.plot(Qvec, S[:,:,3:].sum(axis=1).sum(axis=1))
# plt.grid()
# plt.show()

# import vegas



## SQE by optuna
# enmax = en.max()+0.01

# enarr=en.flatten()
# sarr=S.flatten()
# print(sarr)
# print(f'size before: {sarr.size}')
# # enarr = np.delete(enarr, np.argwhere(sarr < 1e-5))
# # sarr = np.delete(sarr, np.argwhere(sarr < 1e-5))

# print(f'size after: {sarr.size}')

# numbin = 80
# hist, bin_edges = np.histogram(enarr, bins=numbin, range=(0, enmax), density=True, weights=sarr)
# plt.plot(bin_edges[:-1], hist, label='histogram')

# import optuna
# from sklearn.neighbors import KernelDensity

# clf = KernelDensity()
# param_distributions = {
#     'bandwidth': optuna.distributions.FloatDistribution(1e-5, 1e-3, log=True)
# }
# optuna_search = optuna.integration.OptunaSearchCV(clf, param_distributions, n_trials=10)

# optuna_search.fit(enarr[:, np.newaxis], sample_weight=sarr)

# xen=np.linspace(0.0,enmax, numbin)[:, np.newaxis]

# den = optuna_search.score_samples(xen)

# den[den<0]=0
# print(xen.shape, den.shape)
# den=den.reshape(den.size)
# plt.plot(xen, den/den.sum()/(xen[1]-xen[0]), label='normalization')
# plt.plot(xen, den, label='not nor')

## Seq by sklearn kerneldensity
# # from sklearn.neighbors import KernelDensity
# # from sklearn.model_selection import GridSearchCV


# # # use grid search cross-validation to optimize the bandwidth
# # params = {"bandwidth": np.logspace(-5, -3, 5)}
# # grid = GridSearchCV(KernelDensity(), params, n_jobs=8)


# # t1 = time()
# # grid.fit(enarr[:, np.newaxis], sample_weight=sarr)
# # print(f'best bandwidth: {grid.best_estimator_.bandwidth}, histogram bandwidth {enmax/numbin}')

# # # use the best estimator to compute the kernel density estimate
# # kde = grid.best_estimator_

# # # kde = KernelDensity(kernel="gaussian", bandwidth=enmax/numbin/4).fit(enarr[:, np.newaxis], sample_weight=sarr)
# # print(f'Function  executed in {(time()-t1):.4f}s')

# # xen=np.linspace(0.0,enmax, numbin)[:, np.newaxis]
# # t1 = time()
# # den = kde.score_samples(xen)
# # print(f'Function  executed in {(time()-t1):.4f}s')

# # den[den<0]=0
# # print(xen.shape, den.shape)
# # den=den.reshape(den.size)
# # plt.plot(xen, den/den.sum()/(xen[1]-xen[0]))


# plt.legend(loc=0)
# plt.show()