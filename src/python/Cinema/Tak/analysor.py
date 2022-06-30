
import h5py, time, os
from numba import jit, objmode, prange
from Cinema.Interface import *

_autocorrelation = importFunc('autocorrelation', type_voidp, [type_npdbl2d, type_npdbl1d, type_sizet, type_sizet, type_sizet, type_sizet, type_sizet, type_sizet] )

_parFFT = importFunc('parFFT', type_voidp, [type_npcplx2d, type_npcplx2d, type_sizet, type_sizet, type_sizet, type_sizet, type_sizet, type_sizet] )

def parFFT(input, n=None, numcpu=-1):
    dim1=input.shape[0]
    if not n:
        n = input.shape[1]
    out = np.zeros((dim1, n), dtype=np.complex128)
    if numcpu==-1:
        numcpu = os.cpu_count()//2
    _parFFT(input, out, 0, input.shape[0], 1, input.shape[1], n, numcpu)
    return out

def parFFTs(input, offset, spacing, n=None, numcpu=-1):
    dim1=input.shape[0]
    if not n:
        n = input.shape[1]
    out = np.zeros(((dim1-1)//spacing+1, n), dtype=np.complex128)
    if numcpu==-1:
        numcpu = os.cpu_count()//2
    _parFFT(input, out, offset, input.shape[0], spacing, input.shape[1], n, numcpu)
    return out


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


@jit(nopython=True, fastmath=True, parallel=True, inline='always', cache=True)
def diff(arr):
    return arr[:, 1:] - arr[:, :-1]

@jit(nopython=True, fastmath=True, parallel=True, inline='always', cache=True)
def msd(arr):
    return (arr[:, 1:].T - arr[:, 0]).T

def trjdiff(atomictrj, atomoffset, atomPerMolecule, caltype='diff'):
    if atomoffset > atomPerMolecule:
        raise RuntimeError('atomoffset > atomPerMolecule')
    elif atomoffset < 0:
        raise RuntimeError('atomoffset < 0')
    totframe = atomictrj.shape[2]
    totAtom = atomictrj.shape[0]
    loopSize =  totAtom//atomPerMolecule
    res = np.zeros((loopSize*3, totframe-1))
    idx = 0
    for iAtom in range(atomoffset, totAtom, atomPerMolecule):
        if caltype=='diff':
            res[idx:(idx+3), :] = diff(atomictrj[iAtom])
        elif caltype=='msd':
            res[idx:(idx+3), :] = msd(atomictrj[iAtom])
        idx += 3
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
            self.atomid = self.species[0,:self.nAtomPerMole]
            print(f'self.elements {self.elements}, self.nMolecule {self.nMolecule}, self.nAtomPerMole {self.nAtomPerMole}')

            self.trj = hf["particles/all/position/value"][()]
            self.box = hf["particles/all/box/edges/value"][()]
            self.time = hf["particles/all/species/time"][()]
            self.dt = np.diff(self.time).mean()*1e-15
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

def correlation(x, y=None, axis=0, sumOverAxis=None, average=None):
    import numpy
    """Returns the numerical correlation between two signals.

    :param x: the first signal.
    :type x: NumPy array

    :param y: if not None, the correlation is performed between `x` and `y`. If None, the autocorrelation of `x` will be computed.
    :type y: NumPy array or None

    :param axis: the axis along which the correlation will be computed.
    :type axis: int

    :param sumOverAxis: if not None, the computed correlations will be sum over a given axis.
    :type sumOverAxis: int or None

    :param average: if not None, the computed correlations will be averaged over a given axis.
    :type average: int or None

    :return: the result of the numerical correlation.
    :rtype: NumPy array

    :note: The correlation is computed using the FCA algorithm.
    """

    x = numpy.array(x)

    n = x.shape[axis]

    X = numpy.fft.fft(x, 2*n,axis=axis)

    if y is not None:
        y = numpy.array(y)
        Y = numpy.fft.fft(y, 2*n,axis=axis)
    else:
        Y = X

    s = [slice(None)]*x.ndim

    s[axis] = slice(0,x.shape[axis],1)

    corr = numpy.real(numpy.fft.ifft(numpy.conjugate(X)*Y,axis=axis)[s])

    norm = n - numpy.arange(n)

    s = [numpy.newaxis]*x.ndim
    s[axis] = slice(None)

    corr = corr/norm[s]

    if sumOverAxis is not None:
        corr = numpy.sum(corr,axis=sumOverAxis)
    elif average is not None:
        corr = numpy.average(corr,axis=average)

    return corr

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
        _autocorrelation(diff, vdos, 0, diff.shape[0], 1, diff.shape[1], fftsize, numcpu)

        fre = np.fft.fftfreq(vdos.size, self.dt)*2*np.pi
        return fre[:self.nFrame//2], vdos[:self.nFrame//2]

    def msd(self, atomoffset=0, numcpu=-1):
        fftsize = self.nFrame-1
        totAtom = self.nAtom
        atomictrj = self.atomictrj

        #atomid , dim, frame
        '''
        Computes the mean square displacement of a set of coordinates
        :param coords: the set of n coordinates.
        :type coords: (n,3) numpy array
        :return: the mean square displacement.
        :rtype: float
        '''
        coords=np.swapaxes(self.atomictrj[1], 0, 1)

        dsq = np.add.reduce(coords**2,1)

        # sum_dsq1 is the cumulative sum of dsq
        sum_dsq1 = np.add.accumulate(dsq)

        # sum_dsq1 is the reversed cumulative sum of dsq
        sum_dsq2 = np.add.accumulate(dsq[::-1])

        # sumsq refers to SUMSQ in the published algorithm
        sumsq = 2.0*sum_dsq1[-1]

        # this line refers to the instruction SUMSQ <-- SUMSQ - DSQ(m-1) - DSQ(N - m) of the published algorithm
        # In this case, msd is an array because the instruction is computed for each m ranging from 0 to len(traj) - 1
        # So, this single instruction is performing the loop in the published algorithm
        Saabb  = sumsq - np.concatenate(([0.0], sum_dsq1[:-1])) - np.concatenate(([0.0], sum_dsq2[:-1]))

        # Saabb refers to SAA+BB/(N-m) in the published algorithm
        # Sab refers to SAB(m)/(N-m) in the published algorithm
        Saabb = Saabb / (len(dsq) - np.arange(len(dsq)))
        Sab   = 2.0*correlation(coords, axis=0, sumOverAxis=1)

        # The atomic MSD.
        msd = Saabb - Sab
        return np.arange(msd.size)*self.dt, msd

    def saveTrj(self, fileName):
        hf = h5py.File(fileName, 'w')
        # atomid, pos_dim, frameid
        # self.box: frameid, pos_dim
        abox = np.swapaxes(self.box, 0, 1)
        hf['reduced_trj'] = self.atomictrj*2*np.pi/abox
        hf['Q_unit'] = 2*np.pi/self.box.mean()
        hf['dt'] = self.dt
        hf['atomid'] = self.atomid
        hf['atominfo']=np.array([self.nAtom, self.nFrame, self.nMolecule, self.nAtomPerMole], dtype=int)
        hf.close()


def AnaSF2VD(sf):
    vd = AnaVDOS('')
    temp = vars(sf)
    for item in temp:
        setattr(vd, item,  getattr(sf,item))
    vd.unwrap()
    vd.atomictrj = vd.trj
    del vd.trj
    vd.atomictrj = np.swapaxes(vd.atomictrj, 0, 1)
    #swap axes from atomid, frameid, pos_dim to atomid, pos_dim, frameid
    vd.atomictrj = np.swapaxes(vd.atomictrj, 1, 2)
    return vd



@jit(nopython=True, fastmath=True, parallel=True, cache=True)
def scaleQ(exponent, factor, window):
    if window:
        return np.exp(-exponent*1j*factor)*np.kaiser(exponent.shape[2], 20) #*np.hanning(exponent.shape[2])
    else:
        return np.exp(-exponent*1j*factor)


@jit(nopython=True, fastmath=True, parallel=True, cache=True)
def incoherent(b):
    return (b.real*b.real + b.imag*b.imag).sum(axis=0)

@jit(nopython=True, fastmath=True, parallel=True, cache=True)
def coherent(b):
    return b.real.sum(axis=0)**2 + b.imag.sum(axis=0)**2

_coherent_stablesum = importFunc('coherent_stablesum', type_voidp, [type_npcplx2d, type_npdbl1d, type_sizet, type_sizet, type_sizet] )

def coherent_stable(inp, numcpu=8):
    res = np.zeros(inp.shape[1])
    _coherent_stablesum(inp, res,inp.shape[0], inp.shape[1], numcpu)
    return res


class DynamicFactor():
    def __init__(self, inputfile):
        f=h5py.File(inputfile, 'r')
        self.tr = f['reduced_trj'][()]
        self.Q_unit = f['Q_unit'][()]
        self.dt = f['dt'][()]
        self.nAtom, self.nFrame, self.nMolecule, self.nAtomPerMole = f['atominfo'][()]
        self.atomid = f['atomid'][()]
        print(f'self.atomid {self.atomid}, self.tr.shape {self.tr.shape}')
        f.close()

    def calCoherent(self, Q, window=False):
        fftSize = self.tr.shape[2]
        b=np.zeros(fftSize)
        start = time.time()
        inp = scaleQ(self.tr, Q, window)
        inp = inp.reshape(-1, inp.shape[2])
        b = parFFT(inp)
        coh = coherent(b)
        coh = np.fft.fftshift(coh)
        end = time.time()
        print("calCoherent elapsed = %s" % (end - start))
        fre = np.fft.fftshift(np.fft.fftfreq(coh.size, self.dt))*2*np.pi
        return fre, coh

    def calIncoherent(self, Q, window=False):
        fftSize = self.tr.shape[2]
        b=np.zeros(fftSize)

        start = time.time()
        inp = scaleQ(self.tr, Q, window)
        inp = inp.reshape(-1, inp.shape[2])
        b = parFFT(inp)
        inco = incoherent(b)
        inco = np.fft.fftshift(inco)
        end = time.time()
        print("calIncoherent elapsed = %s" % (end - start))
        fre = np.fft.fftshift(np.fft.fftfreq(inco.size, self.dt))*2*np.pi
        return fre, inco
