import numpy as np
import math #isclose
import time
from  Cinema.Prompt.Histogram import Hist2D
from Cinema.Interface.units import hbar

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from Cinema.Interface.units import *
from Cinema.Interface.parallelutil import ParallelHelper
# lattice (Aa), mass (molar mass), pos (fractional coordinate), bc (sqrt(barn))
# qpoints(reduced coordinate), energy(eV ), eigv (unity magnitude), qweight (dimensionless), temperature(kelvin)

import itertools

class PowderHKLIter:
    def __init__(self, lattice_reci, maxQ, step=1):
        self.lattice_reci = lattice_reci
        self.maxQ = maxQ
        qmin1d=np.min([np.linalg.norm(self.lattice_reci[0]),np.linalg.norm(self.lattice_reci[1]),np.linalg.norm(self.lattice_reci[2])])
        maxhkl = int(maxQ/np.ceil(qmin1d))
        # print(f'maxhkl {maxhkl}')
        self.it = itertools.product(range(0, maxhkl+1, step), range(-maxhkl, maxhkl+1, step), range(-maxhkl, maxhkl+1, step))
        self.hkl = None

    def __iter__(self):
        return self

    def __next__(self):
        self.hkl = next(self.it)

        h=int(self.hkl[0])
        k=int(self.hkl[1])
        l=int(self.hkl[2])
        if h==0:
            if k<0:
                next(self) #half a plane
            elif k==0 and l<0: #half an axis, keeping 0,0,0
                next(self)

        # check the Q length
        if np.linalg.norm(np.dot(self.hkl, self.lattice_reci)) > self.maxQ:
            next(self)

        nphkl = np.array(self.hkl)
        if np.allclose(nphkl, np.zeros(3)):
            return {'hkl': nphkl, 'mult': 1}
        else:
            return {'hkl': nphkl, 'mult': 2}


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

        self.vol=np.dot(np.cross(lattice[0],lattice[1]), lattice[2])
        # print('cell volume', self.vol)
        self.xsvolfact = (2*np.pi)**3/self.vol

        self.lattice=lattice
        self.lattice_reci=np.linalg.inv(lattice)*2*np.pi
        self.mass=mass
        self.molarMass=mass/umass
        self.sqMass=np.sqrt(self.mass)
        self.pos=lattice.dot(pos.T).T
        self.bc=bc
        self.fullMess=False
        if np.all(qweight==1.):
            self.fullMess=True
        self.qpoint=np.dot(qpoint, self.lattice_reci)
        self.en=energy
        print(f'negtive energy phonon {len(self.en[self.en<0])}')
        self.en[self.en<0] = -self.en[self.en<0] #fixme: make sure no negtive energy
        self.maxEn = self.en.max()
        self.eigv=eigv
        self.qweight=qweight
        self.temperature=temperature
        self.kt=temperature*boltzmann
        self.detlbal=self.nplus1_down(-self.en)
        self.msd_iso=self.isoMsd() 
        # print('MSD is ' , self.msd_iso)

        

        self.metadata = {
            'lattice': self.lattice,
            'molarMass': self.molarMass,
            'position': self.pos,
            'temperature': self.temperature
            }    #future: may provide API to customize

    def bose(self, en):
        if (en==0.).any():
            raise RuntimeError('input for bose number contains zero')
        n = 1./(np.exp(en/self.kt)-1.)
        return n 
    
    #<n+1>, omega >0, downscattering, good for lowTemp materials where phonons are fewer.
    def nplus1_down(self, en): 
        if (en>0.).any():
            print(en)
            raise RuntimeError('energy for down scattering should be less than zero')
        return self.bose(-en)+1

    def oneplus2n_up(self, en):
        if (en<0.).any():
            raise RuntimeError('energy for up scattering should be greater than zero')
        return 2*self.bose(en)+1


    def calmsd2(self):
        msd=np.zeros([3,3])
        oneplus2n=self.oneplus2n_up(self.en)
        atom=0
        for i in range(self.numQpoint):
          for mode in range(self.nAtom*3):
            #for atom in range(self.nAtom):
            msd+=(np.outer(self.eigv[i,mode, atom], np.conj(self.eigv[i,mode, atom]))).real/self.en[i,mode]*oneplus2n[i,mode]*self.qweight[i]

        msd = msd*hbar*hbar*0.5/self.mass[atom]
        return msd

    def isoMsd(self):
        en, rho = self.dos()
        kernel = self.oneplus2n_up(en)/en*rho
        # coth = 1./ (np.tanh(en/(2*self.kt) ) * en )
        return np.trapz(kernel, en)*0.5*hbar*hbar/self.mass[0]

    def calmsd(self):
        msd=np.zeros([self.nAtom, 3 ,3])
        # print(f'self.qweight.sum {self.qweight.sum()}, {self.eigv.shape}')
        kernel=(self.oneplus2n_up(self.en)/self.en).T*self.qweight
        kernel = kernel.T

        # oneplus2n.shape (132650, 6)
        # self.eigv.shape (132650, 6, 2, 3)

        cutoff = 0.0001
        # cffidx = list(self.en<cutoff)
        # print(cffidx)
        # print(f'num of phonon below threshold {len(self.en[cffidx])}')
    
        for i in range(self.eigv.shape[0]):
            for j in range(self.eigv.shape[1]):
                if self.en[i,j]<cutoff:
                    print('skiped q ', self.qpoint[i])
                    continue

                for a in range(self.nAtom):
                    enj = np.outer( self.eigv[i,j,a], np.conj(self.eigv[i,j,a] ) )
                    msd [a]+= enj.real * kernel[i, j] 

        msd = (msd.T/self.mass*0.5*hbar*hbar).T
        return msd

    def dos(self, bins=100):
        hist=None
        edges=None
        maxEn = self.en.max()
        for i in range(3*self.nAtom):
            h, edges = np.histogram(self.en[:,i], bins=bins, range=[0, maxEn], weights=self.qweight, density=True)
            if hist is not None:
                hist += h
            else:
                hist = h

        center=0.5*(edges[1:]+edges[:-1])
        return center, hist/np.trapz(hist,center)

    def incoherentAppr(self, Q, bins):
        #squires red book 3.136
        # print(f'incoherent msd {self.msd}')
        en, dos = self.dos(bins)
        #for down scattering, en is negtive
        en = -np.flip(en)
        dos = np.flip(dos)
        if isinstance(Q, np.ndarray):
            Q = np.reshape(Q, [Q.size, 1])
     
        common = -1/(4*self.mass[0])*self.bc[0]**2 * Q**2* np.exp(-self.msd_iso*Q**2 )*dos/en*hbar*hbar
        downScat =  common*self.nplus1_down(en)
        return en, downScat


    def calcFormFact(self, Q, eigvecs):
        
        # W = 0.5*Q*Q*u*u = 0.5*Q*Q*MSD
        if self.histparas['msd'] is None:
            Qmag=np.linalg.norm(Q)
            F=(self.bc/self.sqMass*np.exp(-0.5*(self.msd_iso*Qmag*Qmag) )*np.exp(1j*self.pos.dot(Q))*eigvecs.dot(Q)).sum(axis=1) #summing for all atoms
        else:
            w =  -0.5 * np.dot(np.dot(self.histparas['msd'], Q) ,Q)
            F=(self.bc/self.sqMass*np.exp(w + 1j*self.pos.dot(Q))*eigvecs.dot(Q)).sum(axis=1) #summing for all atoms
            # F=(self.bc/self.sqMass*np.exp(w)*np.exp(1j*self.pos.dot(Q))*eigvecs.dot(Q)).sum(axis=1) #summing for all atoms

        #fixme isotropic approximation at the moment
        #F=(self.bc/self.sqMass*np.exp(-0.5*(self.msd.dot(Q))**2 )*np.exp(1j*self.pos.dot(Q))*self.eigv[i].dot(Q)).sum(axis=1)
        return F
    

    def getSw(self, hkl, idx):
        tau=np.dot( np.array(hkl), self.lattice_reci)

        if idx >= self.numQpoint:
            raise RuntimeError('idx >= self.numQpoint')
        
        i = idx        
        if np.allclose(self.qpoint[i], np.zeros(3)):
            raise RuntimeError('calculating sqw at the gamma point ')
        
        Q=self.qpoint[i]+tau
        F = self.calcFormFact(Q, self.eigv[i])

        Smag=0.5*(np.linalg.norm(F)**2)*self.detlbal[i]*hbar*hbar/self.en[i]

        return Smag, np.linalg.norm(Q), self.en[i], 

 
class CalcPowder(CalcBase):
    def __init__(self, lattice, mass, pos, bc, qpoint, energy, eigv, qweight, temperature ):
        super().__init__(lattice, mass, pos, bc, qpoint, energy, eigv, qweight, temperature)

    
    def configHistgrame(self, maxQ, enSize, QSize, extraHistQranage=1., extraHistEnrange = 0.001, msd=None):
        self.histparas = {}
        self.histparas['maxQ']=maxQ
        self.histparas['enSize']=enSize
        self.histparas['QSize']=QSize
        self.histparas['extraHistQranage']=extraHistQranage
        self.histparas['extraHistEnrange']=extraHistEnrange
        self.histparas['msd']=msd

    def getSqw(self):
        if not hasattr(self, 'histparas'):
            raise RuntimeError('configHistgrame() is not called')
        
        q, en_neg, sqw = self.calcPowder(self.histparas['maxQ'])
        en, sqw_inco = self.incoherentAppr(q, en_neg.size)
        return q, en_neg, sqw, sqw_inco


    def savePowerSqw(self, fn, q = None, en_neg = None, sqw= None, sqw_inco = None):
        import h5py
        f0=h5py.File(fn,"w")


        #metadata
        mtd = f0.create_group('metadata')
        for key, value in self.metadata.items():
            mtd.create_dataset(key, data = value)


        if q is None and en_neg is None and sqw is None and sqw_inco is None:
            if not hasattr(self, 'histparas'):
                raise RuntimeError('configHistgrame() is not called')

            q, en_neg, sqw = self.calcPowder(self.histparas['maxQ'])
            en, sqw_inco = self.incoherentAppr(q, en_neg.size)

        ## coherent 
        f0.create_dataset("q", data=q, compression="gzip")
        f0.create_dataset("en", data=en_neg, compression="gzip")
        f0.create_dataset("s", data=sqw, compression="gzip")


        # ## incoherent        
        f0.create_dataset("s_inco", data=sqw_inco, compression="gzip")

        ## expanded sqw

        def expandFromNegtiveAxis(input, axis=0, factor=1.):
            pos = np.flip(input, axis=axis)
            return  np.concatenate((input, pos*factor),axis=axis)

        en_full = expandFromNegtiveAxis(en_neg,factor=-1)
        f0.create_dataset("en_full", data=en_full, compression="gzip")

        en_pos = np.flip(en_neg)*-1
        sqw_full = expandFromNegtiveAxis(sqw, axis=1, factor=np.exp(-np.flip(en_pos)/self.kt))
        f0.create_dataset("sqw_full", data=sqw_full, compression="gzip")

        # print(f'sqw_inco {sqw_inco.shape}, en_neg {en_neg.shape}')

        sqw_full_inco = expandFromNegtiveAxis(sqw_inco, axis=1, factor=np.exp(-np.flip(en_pos)/self.kt))
        f0.create_dataset("sqw_full_inco", data=sqw_full_inco, compression="gzip")
        f0.close()
            

    def calcPowder(self, maxQ):
        it_hkl = PowderHKLIter(self.lattice_reci, maxQ)

        q = None
        en = None
        sqw = None 
        for hkl in it_hkl:
            q_temp, en_temp, sqw_temp = self.calcHKL(hkl)
            if q is None:
                q = q_temp
                en = en_temp
                sqw = sqw_temp
            else:
                sqw += sqw_temp
            
        # per reciprocal space
        qbinmean = np.diff(q).mean() *0.5
        q_edge = np.concatenate(([0], q+qbinmean) )
        q_volume = q_edge**3*np.pi*4/3. # spher volumes
        q_volume_diff = np.diff(q_volume)
        sqw=((sqw.T)/q_volume_diff).T
        # per energy
        sqw/=np.diff(en).mean()

        return  q, en, sqw

    def calcHKL(self, latpnt):
        maxQ = self.histparas['maxQ']
        enSize = self.histparas['enSize']
        QSize = self.histparas['QSize']
        extraHistQranage = self.histparas['extraHistQranage']
        extraHistEnrange = self.histparas['extraHistEnrange']


        # modeWeight = 1./(self.nAtom) 
        # NB: So the xs is in the unit of per atom. Since the eigv from phonopy is already weighted by atom number, so this factor should not be used

        #note negtive energy, for downscattering
        self.hist=Hist2D(0, maxQ + extraHistQranage, QSize, -(self.en.max()+extraHistEnrange), 0, enSize, self.metadata ) 


        # print(f'processing lattice point {latpnt}')
        tau=np.dot(latpnt['hkl'], self.lattice_reci)
        
        for i in range(self.numQpoint):
            if np.allclose(self.qpoint[i], np.zeros(3)):
                continue
            Q=self.qpoint[i]+tau
            Qmag=np.linalg.norm(Q)
            if Qmag > self.hist.xmax:
                continue
            F = self.calcFormFact(Q, self.eigv[i])
            # not devided by the reciprocal volume so the unit is per atoms in the cell
            Smag=0.5*(np.linalg.norm(F)**2)*self.detlbal[i]*self.qweight[i]*hbar*hbar/self.en[i]

            self.hist.fillmany(np.repeat(Qmag,self.nAtom*3), -self.en[i], Smag*latpnt['mult']) #negative energy for downscattering, fill in angular frequency instead of energy       
        return self.hist.xcenter, self.hist.ycenter, self.hist.getWeight()

class CalcBand(CalcBase):
    def __init__(self, lattice, mass, pos, bc, qpoint, energy, eigv, qweight, temperature ):
        super().__init__(lattice, mass, pos, bc, qpoint, energy, eigv, qweight, temperature)

    def calcBand(self):
        pass
