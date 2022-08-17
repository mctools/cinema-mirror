from .units import hbar, boltzmann
from .helper import eKin2k, minMaxQ, expand, angle2Q
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import periodictable as pt
from scipy.interpolate import RectBivariateSpline
import h5py


#example for element_list
#element_list=[{'name': 'H', 'type': 'total', 'number':2}]
#type could be  incoherent, coherent, total
class Sqw:
    def __init__(self, s, q, w, temp, expand_omega=None, element_list=None):
        self.s=np.abs(s)

        self.q=q
        self.w=w
        self.kt=temp*boltzmann
        if expand_omega is not None:
            # self.s = expand(self.s,axis=1, neg_factor=np.exp(-0.5*np.flip(self.w*hbar)/self.kt), pos_factor=np.exp(0.5*self.w*hbar)/self.kt)
            # self.s = expand(self.s,axis=1)
            if expand_omega == 'quantum':
                self.s = expand(self.s,axis=1, neg_factor=np.exp(np.flip(self.w[:-2]*hbar)/self.kt), pos_factor=np.exp(-(self.w*hbar)/self.kt))
                self.w = expand(self.w,neg_factor=-1)
                self.s = (self.s.T*(1./np.trapz(self.s,self.w))).T
            elif expand_omega == 'classic':
                self.s = expand(self.s,axis=1, neg_factor=1, pos_factor=1)
                self.w = expand(self.w,neg_factor=-1)
                self.s = (self.s.T*(1./np.trapz(self.s,self.w))).T
            elif expand_omega == 'first_order':
                self.s = (self.s.T*(0.5/np.trapz(self.s,self.w))).T
                self.s = expand(self.s,axis=1, neg_factor=np.exp(-np.flip(self.w[:-2]*hbar*0.5)/self.kt), pos_factor=np.exp(self.w*hbar*0.5/self.kt))
                self.w = expand(self.w,neg_factor=-1)

        if element_list is not None:
            totnum=0.
            xs=0.
            for element in element_list:
                el_name=element['name'] #e.g. Al, H, O
                type=element['type'] #incoherent, coherent, total
                num=element['number']
                totnum += num
                exs = getattr(getattr(pt, el_name).neutron, type)
                print(f'element xs {exs}')
                xs += exs*num
            scl = xs/(4*np.pi)#/totnum
            self.s *= scl
            print(f'scattering length {scl}')

    def createInteropator(self):
        self.interpobj=RectBivariateSpline(self.q,self.w,self.s)

    def interp(self, qvec, omvec):
        if not hasattr(self, 'interpobj'):
            self.createInteropator()
        return self.interpobj.ev(qvec, omvec)

    def calXSAtFixedAngle(self, enin, enout, angle):
        Q=angle2Q(angle, enin, enout)
        w=(enout-enin)/hbar
        return self.interp(Q,w)*hbar, Q, w


    def plot(self, color_order=1e-10, color_max_scale=1., unitEV=False):
        fig=plt.figure()
        ax = fig.add_subplot(111)
        if unitEV:
            H = self.s.T/hbar
            X, Y = np.meshgrid(self.q, self.w*hbar)
            plt.ylabel('Energy, eV')
        else:
            H = self.s.T
            X, Y = np.meshgrid(self.q, self.w)
            plt.ylabel('Frequency, THz')

        import matplotlib.colors as colors
        pcm = ax.pcolormesh(X, Y, H, cmap=plt.cm.jet,  norm=colors.LogNorm(vmin=H.max()*color_order*color_max_scale, vmax=H.max()*color_max_scale), shading='auto')
        fig.colorbar(pcm, ax=ax)

        plt.xlabel(r'Q ($\AA^{-1}$)')
        plt.grid()
        # plt.show()

    def __str__(self):
        info='Shape of S '+ str(self.s.shape) + '\n'
        info+=f'Shape of Q {self.q.shape}, range [{self.q[0]}, {self.q[-1]}] Aa^-1\n'
        info+=f'Shape of Omega  {self.w.shape}, range [{self.w[0]*1e-12}, {self.w[-1]*1e-12}] THz\n'
        return info

class H5Sqw(Sqw):
    def __init__(self, filename, spath, qpath, wpath, temperature, expand_omega=None, element_list=None):
        f=h5py.File(filename,'r')
        super().__init__(f[spath][()], f[qpath][()], f[wpath][()], f['metadata'][temperature][()], expand_omega, element_list)
        f.close()

class QeSqw(H5Sqw):
    def __init__(self, filename):
        super().__init__( filename, 's', 'q', 'omega', 'temperature')
