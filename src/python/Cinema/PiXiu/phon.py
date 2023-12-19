
import phonopy
from Cinema.PiXiu.io.cell import QeXmlCell
from Cinema.Interface.units import *
import logging
import vegas
import h5py
import multiprocessing as mp

class CohPhon:
    def __init__(self, yamlfile = 'phonopy.yaml', cellQeRelaxXml='out_relax.xml', temperature=300., en_cut = 1e-4) -> None:
        self.ph = phonopy.load(phonopy_yaml=yamlfile, log_level=1, symmetrize_fc=True)
        self.cell = QeXmlCell(cellQeRelaxXml) # this should be changed to the experimental crystal size for production runs
        self.pos=self.cell.reduced_pos.dot(self.cell.lattice)
        self.mass, self.bc, num = self.cell.getAtomInfo()
        self.sqMass = np.sqrt(self.mass)
        self.nAtom = len(self.sqMass)
        logging.info(f'lattice: {self.cell.lattice}')
        logging.info(f'reciprocal lattice: {self.cell.lattice_reci}')

        if any(self.ph.get_unitcell().get_atomic_numbers() != num):
            raise RuntimeError('Different unitcell in the DFT and phonon calculation ')

        self.temperature = temperature
        self.en_cut=en_cut
        meshsize = 100.
        self.ph.run_mesh(meshsize, is_mesh_symmetry=False, with_eigenvectors=True)
        self.ph.run_thermal_displacement_matrices(self.temperature, self.temperature+1, 2, freq_min=0.002)
        # get_thermal_displacement_matrices returns the temperatures and thermal_displacement_matrices
        self.disp = np.copy(self.ph.get_thermal_displacement_matrices()[1][0])
        omega = np.copy(self.ph.get_mesh_dict()['frequencies'])
        self.maxHistEn = omega.max()*THz*2*np.pi*hbar + 0.005 #add 5meV as the energy margin

        import euphonic as eu
        self.eu = eu.ForceConstants.from_phonopy(summary_name='phonopy.yaml')



    def getGammaPoint(self, Qin):
        Q = np.atleast_2d(Qin)
        Qred = self.cell.qabs2reduced(Q)
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
        p = self.eu.calculate_qpoint_phonon_modes(reducedK, use_c=True, asr='reciprocal', n_threads=1)#, useparallel=False)
        return p.frequencies.magnitude*1e-3, p.eigenvectors

    
    def _calcFormFact(self, Qarr, eigvecss, en, tau=None):
        # note, W = 0.5*Q*Q*u*u = 0.5*Q*Q*MSD

        # w =  -0.5 * Q.dot(np.swapaxes( Q.dot(self.disp), 1,2 ) )
        # print(w)

        F = np.zeros((Qarr.shape[0], self.nAtom*3))
        
        for i, (Q, eigvec) in enumerate(zip(Qarr, eigvecss)):
            w =  -0.5 * np.dot(np.dot(self.disp, Q), Q)

            # for testing
            # print('w', w, -0.5*self.disp[0,0,0]*np.linalg.norm(Q)**2)
            # print(self.disp[0])

            #summing for all atoms, F for each mode
            F[i]=np.abs((self.bc/self.sqMass * np.exp(w + 1j*self.pos.dot(Q)) *eigvec.dot(Q)).sum(axis=1))

        # print('eigvec', eigvec[1])
        n = 1./(np.exp(en/(self.temperature*boltzmann))-1.)
        return F*hbar*np.sqrt( (n+1)/en )
    
    def s(self, Qin, reduced=False):
        if reduced:
            Q = self.cell.qreduced2abs(np.asarray(Qin))
        else:
            Q = np.asarray(Qin)

        if Q.ndim==1: 
            Q= np.expand_dims(Q, axis = 0)

        en, eigvec = self._calcPhonon(Q)

        tinyphon = en < self.en_cut # in eV, cufoof small or negtive phonons
        en[tinyphon] = self.temperature*boltzmann # replease small phonon energy to a sensible value to avoid diveded by zero RuntimeWarning
        eigvec[tinyphon] = 1.
        
        Qmag = np.linalg.norm(Q, axis=1)  
        F = self._calcFormFact(Q, eigvec, en)

        # the unit is per atoms by not per reciprocal volume
        Smag = 0.5*F*F
        if tinyphon.any(): # fill zero for small phonons
            Smag[tinyphon] = 0.               

        return Qmag, en, Smag/self.nAtom # per atom  

class kernel(vegas.BatchIntegrand):
    def __init__(self, omegaBin=30, temp=300.) -> None:
        self.calc =  CohPhon(temperature=temp)
        self.omegaRange = [0, self.calc.maxHistEn] 
        self.bin = omegaBin
        self.eventCount=0
        self.recordEvant = False
        self.h5file = None
    
    def enableRecord(self, fn):
        self.recordEvant = True
        id = mp.current_process()
        self.h5file = h5py.File(fn+str(id)+'.h5',"w")

    def disableRecord(self):
        self.recordEvant = False
        self.h5file.close()
        self.h5file = None

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

        Q = np.zeros_like(x)
        Q[:, 0] = cos_theta*sin_phi
        Q[:, 1] = sin_theta*sin_phi
        Q[:, 2] = cos_phi
        Q = (Q.T*r).T
        
        Qmag, en, S = self.calc.s(Q)
        if (S<0.).any():
            print('S<0.', Q, S)
            raise RuntimeError()

        factor = r*r*sin_phi        
        contr = (S.T*factor).T
        contr[np.isnan(contr)]=0.
        I = contr.sum(axis=1)
        
        # return I        
        dI = np.zeros((I.size, self.bin))
        for i in range(I.size):
            dI[i], _ = np.histogram(en[i], bins=self.bin, range=self.omegaRange, weights=contr[i])

        if self.recordEvant:
            QFlatten = np.repeat(Q, 6, axis=0)
            enFlatten = np.atleast_2d(en.flatten()).T
            SFlaten = np.atleast_2d(contr.flatten()).T

            self.h5file.create_dataset(f'event_{self.eventCount}', data=np.concatenate((SFlaten, QFlatten, enFlatten), axis=1), compression="gzip")

        self.eventCount += r.size  
        return dict(I=I, dI=dI)

