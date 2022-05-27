
import h5py, time, os
from numba import jit, objmode, prange
from Cinema.Interface import *

_correlation = importFunc('correlation', type_voidp, [type_npdbl2d, type_npdbl1d, type_sizet, type_sizet, type_sizet, type_sizet, type_sizet, type_sizet] )

@jit(nopython=True, fastmath=True, inline='always', cache=True)
def corrFirstFrame(trj, L, nAtom):
    for iA in prange(nAtom):
        for idim in range(3):
            if trj[iA, idim] < 0.:
                trj[iA, idim] += L[idim]
            elif trj[iA, idim] > L[idim]:
                trj[iA, idim] -= L[idim]
            if trj[iA, idim] < 0 or trj[iA, idim] > L[idim]:
                raise RuntimeError('Corrected postion is still outside the box')


@jit(nopython=True, fastmath=True, inline='always', cache=True)
def corr(trj, L, nAtom, trjold):
    halfL = 0.5*L
    i_L = 1./L
    for iA in prange(nAtom):
        for idim in range(3):
            diff = trj[iA, idim]-trjold[iA, idim]
            if np.abs(diff) > halfL[idim]:
                 trj[iA, idim] -= round(diff*i_L[idim])*L[idim]
            if np.abs(trj[iA, idim]-trjold[iA, idim]) > halfL[idim]:
                raise RuntimeError('Correction wrong')


@jit(nopython=True, fastmath=True, inline='always', cache=True)
def diff(arr):
    sz = arr.size
    res = np.zeros(sz-1)
    for i in prange(sz-1):
        res[i] = arr[i+1]-arr[i]
    return res

def trjdiff(atomictrj, atomoffset, atomPerMolecule):
    if atomoffset > atomPerMolecule:
        raise RuntimeError('atomoffset > atomPerMolecule')
    elif atomoffset < 0:
        raise RuntimeError('atomoffset < 0')
    totframe = atomictrj.shape[2]
    totAtom = atomictrj.shape[0]
    loopSize =  totAtom//atomPerMolecule
    fftsize = totframe*2
    vdos = np.zeros(fftsize, dtype=np.complex128)
    res = np.zeros((loopSize*3, totframe-1))
    idx = 0
    for iAtom in range(atomoffset, totAtom, atomPerMolecule):
        for iDim in range(3):
            res[idx, :] = diff(atomictrj[iAtom, iDim])
            idx += 1
    return res

class Trj():
    def __init__(self, inputfile, unwrap=False):
        if inputfile:
            hf = h5py.File(inputfile, 'r')
            self.species = hf["particles/all/species/value"][()]
            self.nAtom = self.species.shape[1]
            self.nFrame = self.species.shape[0]
            print(self.nAtom, self.nFrame)

            self.elements, counts = np.unique(self.species[0], return_counts=True)
            print(self.elements, counts)
            self.nMolecule = np.gcd.reduce(counts)
            self.nAtomPerMole = self.nAtom//self.nMolecule;
            print(f'self.elements {self.elements}, self.nMolecule {self.nMolecule}, self.nAtomPerMole {self.nAtomPerMole}')

            self.trj = hf["particles/all/position/value"][()]
            self.box = hf["particles/all/box/edges/value"][()]
            self.time = hf["particles/all/species/time"][()]
            print(self.trj.shape)
            print(self.box.shape)
            print(self.time.shape)
            hf.close()

            if unwrap:
                self.unwrap()

    def unwrap(self):
        start = time.time()
        #find atoms outside the box in the first frame
        corrFirstFrame(self.trj[0], self.box[0], self.nAtom)
        #unwrap the rest
        for i in range(1, self.nFrame):
            corr(self.trj[i], self.box[i], self.nAtom, self.trj[i-1])
        end = time.time()
        print("unwrap elapsed = %s" % (end - start))

@jit(nopython=True, fastmath=True, parallel=True, cache=True)
def sqf(upperQ, trj, box, nAtom, nFrame):
    sq = np.zeros(upperQ)
    normFact = 1./(3.* nAtom*nFrame)
    for iQ in prange(upperQ):
        for iFrame in range(nFrame):
            unitQ = 2*np.pi/box[iFrame]
            #atomid, pos_dim
            dotprd = unitQ*trj[iFrame]*(iQ+1)
            sum_cos_term = np.cos(dotprd).sum() #*scattering_length
            sum_sin_term = np.sin(dotprd).sum() #*scattering_length
            sq[iQ] += (sum_cos_term*sum_cos_term) + (sum_sin_term*sum_sin_term)
    return (np.arange(upperQ)+1.)*2*np.pi/box.mean(), sq*normFact

class AnaSFactor(Trj):
    def __init__(self, inputfile):
        super().__init__(inputfile, unwrap=False)

    def getSq(self, upperQ):
        start = time.time()
        q, sq = sqf(upperQ, self.trj, self.box, self.nAtom, self.nFrame)
        end = time.time()
        print("sq elapsed = %s" % (end - start))
        return q, sq


class AnaVDOS(Trj):
    def __init__(self, inputfile):
        super().__init__(inputfile, unwrap=True)
        if inputfile:
            #swap axes from frameid, atomid, pos_dim to atomid, frameid, pos_dim
            self.atomictrj = self.trj
            del self.trj #delete a reference pointed to the same resource
            self.atomictrj = np.swapaxes(self.atomictrj, 0, 1)
            #swap axes from atomid, frameid, pos_dim to atomid, pos_dim, frameid
            self.atomictrj = np.swapaxes(self.atomictrj, 1, 2)

    def vdos_python(self, atomoffset=0): #this method is for unittest only
        fftsize = self.nFrame-1
        totAtom = self.nAtom
        atomictrj = self.atomictrj

        start = time.time()
        diff = trjdiff(atomictrj, atomoffset, self.nAtomPerMole)
        end = time.time()
        print("vdos elapsed = %s" % (end - start))

        vdos = np.zeros(fftsize)
        start = time.time()
        vdos = np.zeros(fftsize)
        for i in range(diff.shape[0]):
            temp = np.fft.fft(diff[i], n=fftsize)
            vdos += np.abs(temp)**2
        end = time.time()
        print("vdos_python diff elapsed = %s" % (end - start))
        return vdos[:self.nFrame//2]

    def vdos(self, atomoffset=0, numcpu=-1):
        fftsize = self.nFrame-1
        totAtom = self.nAtom
        atomictrj = self.atomictrj

        start = time.time()
        diff = trjdiff(atomictrj, atomoffset, self.nAtomPerMole)
        end = time.time()
        print("vdos diff elapsed = %s" % (end - start))

        vdos = np.zeros(fftsize)
        if numcpu==-1:
            numcpu = os.cpu_count()//2
        #atom trajectories are piecked by trjdiff already
        print(f'trj diff shape {diff.shape[0]} {diff.shape[1]} ')
        _correlation(diff, vdos, 0, diff.shape[0], 1, diff.shape[1], fftsize, numcpu)
        return vdos[:self.nFrame//2]

    def saveTrj(self, fileName):
        hf = h5py.File(fileName, 'w')
        hf['trj'] = self.atomictrj
        hf.close()


def AnaSF2VD(sf):
    vd = AnaVDOS('')
    vd.species = sf.species
    vd.nAtom = sf.nAtom
    vd.nFrame = sf.nFrame
    vd.trj = sf.trj
    vd.box = sf.box
    vd.time = sf.time
    vd.nMolecule = sf.nMolecule
    vd.nAtomPerMole = sf.nAtomPerMole
    vd.unwrap()
    vd.atomictrj = vd.trj
    del vd.trj
    vd.atomictrj = np.swapaxes(vd.atomictrj, 0, 1)
    #swap axes from atomid, frameid, pos_dim to atomid, pos_dim, frameid
    vd.atomictrj = np.swapaxes(vd.atomictrj, 1, 2)
    return vd
