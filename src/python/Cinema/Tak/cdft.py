from Cinema.Tak import *
from scipy.interpolate import splev, splrep
import h5py
import matplotlib.pyplot as plt

from Cinema.Interface import units
from Cinema.Interface.helper import getOmegaFromTime, convOmT, takfft
from Tak import takconv
from .FunctionXY import FunctionXY

class DensityOfState():
    def __init__(self, fre, dos):
        if np.isnan(dos).any():
            raise RuntimeError("vdos has nan!!!")
        self.dos = dos
        self.fre = fre
        print(f'self.fre size {self.fre.size}')
        scale = np.trapz(dos,fre)
        self.dos /= scale

    def cutoff(self, energy=None, fre=None):
        # if (energy and fre) is None:
        #     raise RuntimeError('Either energy or freqency should be defined')
        if energy is not None:
            fre = energy/units.hbar
        idx = np.searchsorted(self.fre, fre)
        self.fre = self.fre[:idx]
        self.dos = self.dos[:idx]
        self.dos -= self.dos[-1]
        scale = np.trapz(self.dos,self.fre)
        self.dos /= scale


    def plot(self):
        import matplotlib.pyplot as plt
        plt.plot(self.fre, self.dos)
        plt.show()

    def thicken(self, times, linear=True, update=True):
        if times == 1:
            return self.fre, self.dos
        #interp vdos for x times
        if linear:
            spline_fre=np.linspace(self.fre[0], self.fre[-1], self.fre.size*times)
        else:
            spline_fre=np.logspace(np.log10(self.fre[0]), np.log10(self.fre[-1]), self.fre.size*times)
        spl = splrep(self.fre, self.dos)
        spline_dos = splev(spline_fre, spl)
        if update:
            self.fre = spline_fre
            self.dos = spline_dos
        return spline_fre, spline_dos

    def cal_gamma_filon(self,tarr, mass, temp):
        #calculate second or third part of time vec, using filon or interp n times filon
        x = self.fre#.copy()
        y = self.dos#.copy()

        l_cls,l_real,l_imag = tak_cal_limit_integral(x,y,tarr, mass, temp)
        i_cls,i_real,i_imag = tak_cal_filon(x[1:],y[1:],tarr, mass, temp)
        ans_cls=l_cls+i_cls
        ans_real=l_real+i_real
        ans_imag=l_imag+i_imag
        return ans_cls,ans_real,ans_imag

class Gamma(DensityOfState):
    def __init__(self, filename, wpath, vpath, mass):
        f=h5py.File(filename,'r')
        super().__init__(f[wpath][()], f[vpath][()])
        f.close()
        self.mass = mass
        self.deltaf = np.diff(self.fre).mean()

    def saveGamma(self, filename, temp, thicken, tarr=None):
        if tarr is None:
            _, tarr = convOmT(self.fre.size*4, self.deltaf, False)

        print(f'savGamma, tarr size {tarr.size}')
        self.thicken(thicken)
        g_cls, g_real, g_imag = self.cal_gamma_filon(tarr, self.mass, temp)
        f0=h5py.File(filename,"w")
        f0.create_dataset("time_vec", data = tarr, compression="gzip")
        f0.create_dataset("gamma_qtm_real", data=g_real*1e14, compression="gzip")
        f0.create_dataset("gamma_qtm_imag", data=g_imag*1e14, compression="gzip")
        f0.create_dataset("gamma_cls", data=g_cls*1e14, compression="gzip")
        f0.close()
        return tarr, g_cls, g_real, g_imag

def expandGammaFromH5(filename):
    f=h5py.File(filename,'r')
    t_old=f["time_vec"][()]
    g_cls = f['gamma_cls'][()]
    g_qtm_r=f["gamma_qtm_real"][()]
    g_qtm_i=f["gamma_qtm_imag"][()]
    f.close()
    #print(t_old.size)
    t_neg = -np.flip(t_old)
    # print("t neg=",t_neg)
    time=np.concatenate((t_neg,t_old[1:]))
    g_neg = np.flip(g_cls)
    gamma_cls=np.concatenate((g_neg,g_cls[1:]))
    g_neg = np.flip(g_qtm_r)
    gamma_qtm_r=np.concatenate((g_neg,g_qtm_r[1:]))
    g_neg = -np.flip(g_qtm_i)
    gamma_qtm_i=np.concatenate((g_neg,g_qtm_i[1:]))

    gamma_qtm=np.zeros(time.size,dtype=complex)
    for i in range(time.size):
        gamma_qtm[i]=complex(gamma_qtm_r[i],gamma_qtm_i[i])
    # gamma_cls,gamma_qtm = self.getSplineGamma(time,t_old,g_cls,g_qtm_r,g_qtm_i)
    return time,gamma_cls,gamma_qtm

class HDRFT():
    def __init__(self, tVec, exponent):
        self.dt = tVec[1]-tVec[0]
        if tVec[0]+tVec[-1] > 2*self.dt:
            raise RuntimeError("tVec[0]+tVec[-1] > 2*dt  ")
        if not np.all(np.diff(tVec)) > 0.:
            raise RuntimeError("tVec is expected to be monotonically increasing ")
        self.t=tVec
        self.deltaf, self.freq = getOmegaFromTime(self.t.size, self.dt)
        self.exponent = exponent

    def getExponent(self):
        return self.exponent

    def calcFFT(self,window=None):
        y=np.exp(self.getExponent())
        if window is not None:
            if window=="kaiser":
                y*=np.kaiser(y.size,20)
            if window=="hanning":
                y*=np.hanning(y.size)
        directFFT = np.abs(takfft(y, self.dt))
        directFre = self.freq
        return directFre, directFFT

    def calHdRFT(self, order, firstOrderCutoff , distortFact,
                asymExponent, offset=0,window=None,isNormalise=False, xBoundary=10./units.hbar,autodistort=False):
        hdrftFre = self.freq
        ft=self.getExponent()
        x=ft.min()+offset
        rt=ft-x
        # print(f"x={x}")
        r0=np.interp(0., self.t, rt)
        f0=np.ones(self.t.size)
        f1=rt/r0
        f0*=np.kaiser(f0.size,20)
        f1*=np.kaiser(f1.size,20)
        g0=takfft(f0, self.dt)
        g1=takfft(f1, self.dt)

        deltaOmega = abs(self.freq[-1]-self.freq[0])/(self.freq.size-1)
        totalxy = FunctionXY(self.freq, np.abs(r0*g1+g0), distortFact,asymExponent, autodistort)

        g1=takfft(f1*np.kaiser(f1.size,20), self.dt)
        g1xy = FunctionXY(self.freq, np.copy(g1), distortFact,asymExponent, autodistort)
        gnxy=FunctionXY(self.freq, np.copy(g1), distortFact, asymExponent, autodistort)

        # if isNormalise:
        g1xy.normalise()
        gnxy.normalise()

        totalxy.crop(-firstOrderCutoff, firstOrderCutoff)
        g1xy.crop(-firstOrderCutoff, firstOrderCutoff)
        gnxy.crop(-firstOrderCutoff, firstOrderCutoff)
        g1xy.distort()
        g1xy.flipNeg2Pos()


        gnxy.distort()
        gnxy.flipNeg2Pos()

        coef = r0

        for n in range(2, order+1):
            gnxy = takconv(gnxy, g1xy)
            gnxy.flipNeg2Pos()
            gnxy.restort()
            # if isNormalise:
            # gnXYRecoveredAear = np.trapz(np.abs(gnxy._Curve__y), gnxy.x)
            # print(f'gnxy order {n}, recovered area {gnXYRecoveredAear}')
            # #making sure the area is always unity
            gnxy.normalise()

            gnXYRecovered=np.abs(gnxy.y)
            gnXYRecoveredAear = np.trapz(gnXYRecovered, gnxy.x)
            print(f'gnxy order {n}, recovered area {gnXYRecoveredAear}')
            # if gnXYRecoveredAear>1.1 or gnXYRecoveredAear< 0.9:
            #     plt.figure()
            #     plt.semilogy(gnxy.x*units.hbar, gnXYRecovered, label= f'recovered, order={n} ')
            #     gnxy.distort()
            #     plt.semilogy(gnxy.x*units.hbar, np.abs(gnxy.y), label= f'distorted, order={n} ')
            #     plt.title('Debug info, recovered result is not close to unity')
            #     plt.legend()
            #     plt.show()
            #     raise RuntimeError('recovered result is not close to unity')

            if autodistort:
                gnxy.calDistortFact()

            coef *= r0/n
            gnxy.scaleY(coef)
            totalxy.accumulate(gnxy)
            gnxy.scaleY(1./coef)

            gnxy.distort()


            if isNormalise:
                gnxy.scaleY(1./gnXYRecoveredAear)

            if gnxy.x[-1]> xBoundary:
                gnxy.crop(-xBoundary,xBoundary)
        totalxy.scaleY(np.exp(x))
        totalxy.flipNeg2Pos()
        if isNormalise:
            totalxy.normalise()

        return totalxy.x, totalxy.y
