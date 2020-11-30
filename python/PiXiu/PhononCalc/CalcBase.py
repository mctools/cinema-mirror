import numpy as np
import math #isclose
import time
from PiXiu.Common.Units import *

# lattice (Aa), mass (atomic mass), pos (fractional coordinate), bc (sqrt(barn))
# qpoints(reduced coordinate), energy(eV ), eigv (unity magnitude), qweight (dimensionless), kt(eV)

class CalcBase:
    def __init__(self, lattice, mass, pos, bc, qpoint, energy, eigv, qweight, kt ):
        if mass.ndim!=1:
            raise ValueError('Mass is not 1D array')

        self.nAtom = len(mass)

        if lattice.shape!=(3, 3):
            raise ValueError('Wrong lattice matrix')

        if pos.shape!=(self.nAtom, 3):
            raise ValueError('Expected shape of the atom position matrix is {}, but it is {}'.format( (self.nAtom, 3), (pos.shape)))

        if bc.shape!=(self.nAtom, ):
            raise ValueError('Scattering length array in wrong size')

        if qweight.ndim!=1:
            raise ValueError('qweight is not 1D array')
        self.numQpoint = len(qweight)

        if qpoint.shape!=(self.numQpoint, 3):
            raise ValueError('Expected shape of the qpoints matrix is {}, but it is {}'.format( (self.numQpoint, 3), (qpoint.shape)))

        if eigv.shape!=(self.numQpoint, self.nAtom*3, self.nAtom, 3):
            raise ValueError('Expected shape of the eigenvector is {}, but it is {}'.format( (self.numQpoint, self.nAtom*3, self.nAtom, 3), (eigv.shape)))

        self.lattice_reci=np.linalg.inv(lattice)*2*np.pi
        self.mass=mass*umass
        self.sqMass=np.sqrt(self.mass)
        self.pos=lattice.dot(pos.T).T
        self.bc=bc
        self.qpoint=np.dot(qpoint, self.lattice_reci)
        self.en=energy
        self.maxEn = self.en.max()
        self.eigv=eigv
        self.qweight=qweight
        self.kt=kt
        self.md=np.sqrt(self.calmsd()) #mean displacement in Aa
        self.bose=self.nplus1(self.en)

        if not math.isclose(self.qweight.sum(), 1.0, rel_tol=1e-10):
            raise ValueError('phonon total qweight is not unity {}'.format( self.qweight.sum() ) )

    def calmsd2(self):
        msd=np.zeros([3,3])
        oneplus2n=self.oneplus2n(self.en)
        atom=0
        for i in range(self.numQpoint):
          for mode in range(self.nAtom*3):
            #for atom in range(self.nAtom):
            msd+=(np.outer(self.eigv[i,mode, atom], np.conj(self.eigv[i,mode, atom]))).real/self.en[i,mode]*oneplus2n[i,mode]*self.qweight[i]

        msd = msd*hbar*hbar*0.5/self.mass[atom]
        return msd

    def calmsd(self):
        msd=np.zeros([3,self.nAtom])
        oneplus2n=self.oneplus2n(self.en)
        for i in range(self.numQpoint):
           msd+=(np.abs(self.eigv[i,:]* np.conj(self.eigv[i, :]) ).T/self.en[i]*oneplus2n[i]*self.qweight[i]).sum(axis=2)
        msd*=0.5/self.mass*hbar*hbar
        return msd.T

    def dos(self):
        hist=None
        edges=None
        maxEn = self.en.max()
        for i in range(3*self.nAtom):
            h, edges = np.histogram(self.en[:,i], bins=100, range=[0, maxEn], weights=self.qweight, density=True)
            if hist is not None:
                hist += h
            else:
                hist = h

        center=0.5*(edges[1:]+edges[:-1])
        return center, hist/np.trapz(hist,center)

    def isoMsd(self):
        en, rho = self.dos()
        return np.trapz( (1./(np.tanh(en/(2*self.kt) ) * en )* rho), en)*0.5*hbar*hbar/self.mass[0]


    #<n+1>, omega >0, downscattering
    def nplus1(self, en): #fixme: nan
        return 1./(1- np.exp(-en/self.kt))

    def oneplus2n(self, en):#fixme: nan
        invkt = 1./self.kt
        return 1/np.tanh(0.5*en*invkt)


    def calcSqw(self, hkl, maxQ, enSize, QSize):
        #S=np.array([self.nAtom*3*self.numQpoint, 3])
        S=[]
        tau=np.dot(hkl,self.lattice_reci)
        modeWeight = 1./(self.nAtom) #fixme: So the xs is in the unit of per atom? but the  eigv is already weighted
        for i in range(self.numQpoint):
            Q=self.qpoint[i]+tau
            Qmag=np.linalg.norm(Q)
            F=(self.bc/self.sqMass*np.exp(-0.5*(self.md.dot(Q))**2 )*self.eigv[i].dot(Q)).sum(axis=1)
            Smag=np.abs(F*F)*self.bose[i]*self.qweight[i]*hbar*modeWeight/self.en[i]
            for n in range(self.nAtom*3):
               S.append([Smag[n], Qmag, self.en[i][n]])

            # S[(i*self.nAtom*3):((i+1)*self.nAtom*3), 0] = Smag
            # S[(i*self.nAtom*3):((i+1)*self.nAtom*3), 1] = Qmag
            # S[(i*self.nAtom*3):((i+1)*self.nAtom*3), 2] = en[i]

        S=np.array(S)

        H, xedges, yedges = np.histogram2d(S[:,1], S[:,2], range= [[0, maxQ], [0,self.maxEn]], bins=[QSize, enSize], weights=S[:,0])
        return H, xedges, yedges

    def show(self, H, xedges, yedges):
        import matplotlib.pyplot as plt
        fig=plt.figure()
        ax = fig.add_subplot(111)
        H = H.T

        X, Y = np.meshgrid(xedges, yedges)
        import matplotlib.colors as colors
        pcm = ax.pcolormesh(X, Y, H, cmap=plt.cm.jet,  norm=colors.LogNorm(vmin=H.max()*1e-4, vmax=H.max()),)
        fig.colorbar(pcm, ax=ax)
        plt.xlabel('Q, Aa^-1')
        plt.ylabel('energy, eV')
        plt.show()
