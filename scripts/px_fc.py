#!/usr/bin/env python3

import numpy as np
from Cinema.PiXiu.io.cell import QeXmlCell
from Cinema.Interface.units import *
import euphonic as eu
np.set_printoptions(precision=4, suppress=False)

au=0.529177248994098*aa
Ry = 13.605662285137*eV
RyPerAa2 = Ry/au/au

cell = QeXmlCell('out_relax.xml')
# print(cell.lattice, cell.lattice.dot([0.1,0,0]))


data = np.loadtxt('data_444')
# print(data)

AMU_SI           = 1.66053906660E-27 # ! Kg
ELECTRONMASS_SI  = 9.1093837015E-31 #  ! Kg

AMU_AU           = AMU_SI / ELECTRONMASS_SI
AMU_RY           = AMU_AU / 2.0

mass = 25598.367262405940/AMU_RY

mass = 28.0855
print('mass', mass)

nAtom = 2
nPair = nAtom*nAtom
nIdx = 9
n1 = 4
n2 = 4
n3 = 4
supCell = np.array([n1, n2, n3])
halfSupCell = supCell/2.

nUCell = n1*n2*n3

# https://cahmd.gitlab.io/cahmdweb/cahmdwiki/info/Quantum_Espresso/QuEs_calculate_phonons.html
# http://www.democritos.it/pipermail/pw_forum/2005-April/002408.html
# http://www.democritos.it/pipermail/pw_forum/2009-August/013613.html 
# > 4 4 4 - you used 4x4x4 q-points to generate FC
# > 1 3 1 2 - matrix alhpa, matrix beta, 1st atom, 2nd atom
# > 4 3 2 0.1234,  cell index and force 
# 
#  F_{atom1, atom2}^{idx_y, idx_x}
# 

def cal(Q):
    dynmat = np.zeros((nAtom*3, nAtom*3),dtype=complex)

    line = 0
    for i in range(nAtom*nAtom*9):
        line = (nUCell+1)*i
        alpha, beta, atomA, atomB = np.rint(data[line]-1) # fortran indexing
        # print('abcd', line, atomA, atomeB, alpha, beta)

        line += 1
        for pair in range(nUCell):
            cell_redpos = data[line, 0:3] - 1 # 1,1,1 is the origin
            ref_idx = cell_redpos > halfSupCell
            if (ref_idx).any():
                # print('cell_redpos', cell_redpos)
                cell_redpos[ref_idx] -= supCell[ref_idx]
                # print('after cell_redpos', cell_redpos)
    # addphase = np.exp(1j*cell.lattice.dot(np.array([1,1,1])).dot(Q))
    # dynmat *= addphase

        
            r = cell.lattice.dot(cell_redpos)
            
            phase = np.exp(-1j*r.dot(Q))
            contr = data[line, 3] *  phase
            
            dynmat[int(atomA*3+alpha), int(atomB*3+beta)] += contr
            
            # print(data[line])
            line += 1
        # print('section ended with ', data[line-1])

    dynmat *= 1./mass/np.pi/np.pi
    dynmat = dynmat+np.conj(dynmat.T)
    # addphase = np.exp(1j*cell.lattice.dot(np.array([1,1,1])).dot(Q))
    # dynmat *= addphase


    # print(np.abs(dynmat))
    # dynmat=dynmat.T.reshape(nAtom*3, nAtom*3)
    val, eigv = np.linalg.eigh(dynmat)
    return np.sign(val)*np.sqrt(np.abs(val)), eigv

# #######################################

Qred = np.array([[0,0,0]])
Q = cell.qreduced2abs(Qred)

px_en, px_eigv = cal(Q[0])

fc = eu.ForceConstants.from_phonopy(summary_name='phonopy.yaml')
p = fc.calculate_qpoint_phonon_modes(Qred)
eu_en = p.frequencies.magnitude*1e-3
eu_eigv = p.eigenvectors

print(px_en) 
print(eu_en)

print(np.abs(px_eigv.reshape((-1,2,3)))) 
print(np.abs(eu_eigv))


#######################################
loop = 20


GAMMA = [0.0, 0.0, 0.0]
X = [0.5, 0.0, 0.5]
U = [0.625, 0.25, 0.625]

p1 = np.linspace(GAMMA[0], X[0], loop)
Qred = np.array([p1, np.zeros_like(p1), p1]).T

# p2_x = np.linspace(X[0], U[0], loop)
# p2_y = np.linspace(X[1], U[1], loop)
# p2_z = np.linspace(X[2], U[2], loop)
# Qred = np.array([p2_x, p2_y, p2_z]).T


Q = cell.qreduced2abs(Qred)


fre = np.zeros((loop, 6))

for i in range(loop):
    fre[i], _ = cal(Q[i])


# print('ratio', 14.866602/-0.416592, vec[-1]/vec[0] )
# ref = 14.866602*THz*2*np.pi*hbar
# print(ref, ref/vec[-1], vec[-1]/ref )

import matplotlib.pyplot as plt
plt.plot(fre)

fc = eu.ForceConstants.from_phonopy(summary_name='phonopy.yaml')
p = fc.calculate_qpoint_phonon_modes(Qred)
euen = p.frequencies.magnitude*1e-3

plt.figure()
plt.plot(euen)
plt.show()









## Cal XS in si
# import euphonic as eu
# import numpy as np
# planck = 4.13566769692386e-15  #(source: NIST/CODATA 2018)
# hbar = planck*0.5/np.pi  #[eV*s]6.582119569509068e-16
# s=1.
# THz=1e12/s


# print(eu.__version__)
# Q = np.array([[0.1,0.1,0.1], [1.1,0.1,0.1]])

# fc = eu.ForceConstants.from_phonopy(summary_name='phonopy.yaml')
# p = fc.calculate_qpoint_phonon_modes(Q)
# print(type(p.frequencies),)

# # eu.pint.util.Quantity
# # raise RuntimeError()
# print(p.eigenvectors.shape)
# # print((np.abs(p.eigenvectors)**2).sum(axis=1))

# for ei in p.eigenvectors:
#     print('mag', ei.shape, np.linalg.norm(ei, axis=-1)**2)

# import phonopy

# ph = phonopy.load(phonopy_yaml='phonopy.yaml', log_level=1, symmetrize_fc=True)

# ph.run_qpoints(Q, with_eigenvectors=True)  

# res = ph.get_qpoints_phonon()
# # ph.write_hdf5_qpoints_phonon()
# nAtom = fc.crystal.n_atoms

# en = res[0]*THz*2*np.pi*hbar*1e3
# eigv = res[1].reshape((-1, nAtom*3, nAtom, 3)) # en, eigenvector


# # for ei in eigv:
# #     print('mag', ei.shape, np.linalg.norm(ei, axis=-1)**2)


# for e1, e2 in zip(p.frequencies, en):
#     for a1, a2 in zip(e1,e2):
#        print('eu', a1, '\npy', a2, '\n\n')

# # for e1, e2 in zip(p.eigenvectors, eigv):
# #     for a1, a2 in zip(e1,e2):
# #        for b1, b2 in zip(a1, a2):
# #            print('eu', np.abs(b1), (np.abs(b1)**2).sum(), '\npy', np.abs(b2), (np.abs(b2)**2).sum(), '\n\n')


