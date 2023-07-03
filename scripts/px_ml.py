#!/usr/bin/env python3

import numpy as np
from Cinema.PiXiu.io.cell import QeXmlCell
import phonopy
import matplotlib.pyplot as plt

# np.set_printoptions(suppress=True)

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

        if any(self.ph.get_unitcell().get_atomic_numbers() != num):
            raise RuntimeError('Different unitcell in the DFT and phonon calculation ')

        self.temperature = temperature
        meshsize = 100.
        self.ph.run_mesh(meshsize, is_mesh_symmetry=False, with_eigenvectors=True)
        self.ph.run_thermal_displacement_matrices(self.temperature, self.temperature+1, 2)
        # get_thermal_displacement_matrices returns the temperatures and thermal_displacement_matrices
        self.disp = self.ph.get_thermal_displacement_matrices()[1][0]
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
       
    def _calcPhonon(self, k, reduced=False):
        if reduced:
            self.ph.run_qpoints(k, with_eigenvectors=True)
        else:
            self.ph.run_qpoints(self.cell.qabs2reduced(k), with_eigenvectors=True)

        res = self.ph.get_qpoints_phonon()
        # self.ph.write_hdf5_qpoints_phonon()
        nAtom = len(self.cell.element)
        return res[0]*THz*2*np.pi*hbar, res[1].reshape((-1, nAtom*3, nAtom, 3)) # en, eigenvector

    
    def _calcFormFact(self, Q, tau, eigvecs):
        # note, W = 0.5*Q*Q*u*u = 0.5*Q*Q*MSD
        w =  -0.5 * np.dot(np.dot(self.disp, Q), Q)

        # for testing
        # print('w', w, -0.5*self.disp[0,0,0]*np.linalg.norm(Q)**2)
        # print(self.disp[0])

        #summing for all atoms, F for each mode
        F=(self.bc/self.sqMass*np.exp(w + 1j*self.pos.dot(Q))*eigvecs.dot(Q)).sum(axis=1) 

        return F


    def s(self, hkl, qin, reduced=True):
        tau = self.cell.lattice_reci.dot(np.asarray(hkl))
        
        if reduced:
            q = self.cell.qreduced2abs(np.asarray(qin))
        else:
            q = np.asarray(qin)

        Q = q + tau        
        en, eigvec = self._calcPhonon(Q,  reduced)

        if np.allclose(Q, np.zeros(3)):
            print('hit gamma')
            return None
        
        F = self._calcFormFact(Q, tau, eigvec[0])
        # print(F)
        # not devided by the reciprocal volume so the unit is per atoms in the cell
        n = 1./(np.exp(en/(self.temperature*boltzmann))-1.)

        Smag = 0.5*(np.linalg.norm(F)**2)*hbar*hbar/en* (n + 1)
        return np.linalg.norm(Q) , en, Smag


calc = CohPhon()

Q, en, S = calc.s([-1,0,0], [-0.5,-0.5,-0.5], reduced=True)
print(Q, en, S )

Q, en, S = calc.s([1,0,0], [0.5,0.5,0.5], reduced=True)
print(Q, en, S )


# Q, en, S = calc.s([0,0,0], [0.01913645, 0.01913645, 0.01913645], reduced=False)
# print(Q, en, S )