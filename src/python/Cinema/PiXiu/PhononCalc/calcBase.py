import numpy as np
import math #isclose
import time
from  Cinema.Prompt.Math.Hist import Hist2D
from Cinema.Interface.units import hbar

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from Cinema.Interface.units import *

# lattice (Aa), mass (atomic mass), pos (fractional coordinate), bc (sqrt(barn))
# qpoints(reduced coordinate), energy(eV ), eigv (unity magnitude), qweight (dimensionless), temperature(kelvin)

class CalcBase:
    def __init__(self, lattice, mass, pos, bc, qpoint, energy, eigv, qweight, temperature ):
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

        self.lattice=lattice
        self.lattice_reci=np.linalg.inv(lattice)*2*np.pi
        self.mass=mass*umass
        self.reducedMass=mass
        self.sqMass=np.sqrt(self.mass)
        self.pos=lattice.dot(pos.T).T
        self.bc=bc
        self.fullMess=False
        if np.all(qweight==1.):
            self.fullMess=True
        self.qpoint=np.dot(qpoint, self.lattice_reci)
        self.en=energy
        self.maxEn = self.en.max()
        self.eigv=eigv
        self.qweight=qweight
        self.kt=temperature*boltzmann
        self.bose=self.nplus1(self.en)
        self.msd=self.isoMsd() #fixme: calmsd method returns unequal msd even for cubic lattice, bug to be fixed use isotropic model for now.

        if not math.isclose(self.qweight.sum(), 1.0, rel_tol=1e-10):
            raise ValueError('phonon total qweight is not unity {}'.format( self.qweight.sum() ) )

    #<n+1>, omega >0, downscattering
    def nplus1(self, en): #fixme: nan
        return 1./(1- np.exp(-en/self.kt))

    def oneplus2n(self, en):#fixme: nan
        invkt = 1./self.kt
        return 1/np.tanh(0.5*en*invkt)


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

    def calcFormFact(self, Q, eigvecs):
        Qmag=np.linalg.norm(Q)
        F=(self.bc/self.sqMass*np.exp(-0.5*(self.msd*Qmag*Qmag) )*np.exp(1j*self.pos.dot(Q))*eigvecs.dot(Q)).sum(axis=1)
        #fixme isotropic approximation at the moment
        #F=(self.bc/self.sqMass*np.exp(-0.5*(self.msd.dot(Q))**2 )*np.exp(1j*self.pos.dot(Q))*self.eigv[i].dot(Q)).sum(axis=1)
        return F

class CalcPowder(CalcBase):
    def __init__(self, lattice, mass, pos, bc, qpoint, energy, eigv, qweight, temperature ):
        super().__init__(lattice, mass, pos, bc, qpoint, energy, eigv, qweight, temperature)

    def calcPowder(self, maxQ, enSize, QSize, extraHistQranage=1., extraHistEnrange = 0.001, step=1):
        qmin1d=np.min([np.linalg.norm(self.lattice_reci[0]),np.linalg.norm(self.lattice_reci[1]),np.linalg.norm(self.lattice_reci[2])])
        maxhkl = np.int(maxQ/np.ceil(qmin1d))

        hist=Hist2D(0, maxQ + extraHistQranage, QSize, -(self.en.max()+extraHistEnrange)/hbar, 0, enSize ) #note negtive energy, for  downscattering, y-axis is in THz

        for h in range(0,maxhkl+1,step):  # half a space
                for k in range(-maxhkl,maxhkl+1,step):
                    hkllist=[]
                    for l in range(-maxhkl,maxhkl+1,step):
                        if h==0:
                            if k<0:
                                continue #half a plane
                            elif k==0 and l<0: #half an axis, keeping 0,0,0
                                continue

                        hkl=np.array([h,k,l])
                        if np.linalg.norm(np.dot(hkl,self.lattice_reci)) > maxQ:
                            print('skipped hkl', (h,k,l))
                            continue

                        print('processing hkl', (h,k,l))

                        if not(h==0 and k==0 and l==0):
                            self.calcHKL(hkl, hist)
                        else:
                            self.calcHKL(hkl, hist, 2)
        return hist

    def calcHKL(self, hkl, hist, hklweight=1.):
        #S=np.array([self.nAtom*3*self.numQpoint, 3])
        tau=np.dot(hkl,self.lattice_reci)
        modeWeight = 1./(self.nAtom) #fixme: So the xs is in the unit of per atom? but the  eigv is already weighted
        for i in range(self.numQpoint):
            Q=self.qpoint[i]+tau
            Qmag=np.linalg.norm(Q)
            if Qmag > hist.xmax:
                continue
            F = self.calcFormFact(Q, self.eigv[i])
            Smag=modeWeight*(np.linalg.norm(F)**2)*self.bose[i]*self.qweight[i]*hbar/self.en[i]
            hist.fillmany(np.repeat(Qmag,self.nAtom*3), -self.en[i]/hbar, Smag*hklweight*hbar) #negative energy for downscattering, fill in angular frequency instead of energy

class CalcBand(CalcBase):
    def __init__(self, lattice, mass, pos, bc, qpoint, energy, eigv, qweight, temperature ):
        super().__init__(lattice, mass, pos, bc, qpoint, energy, eigv, qweight, temperature)

    def calcBand(self):
        pass
