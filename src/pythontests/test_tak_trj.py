#!/usr/bin/env python3

import numpy as np
from Cinema.Interface.Utils import findData
from Cinema.Tak.analysor import AnaVDOS, AnaSFactor, AnaSF2VD
from Cinema.Tak.analysor import DynamicFactor
from Cinema.Interface import units


dumpfile=findData('md/dump_h5md.h5')
s = AnaSFactor(dumpfile)
q, sq = s.getSq(10)

np.set_printoptions(precision=16)
print(q, sq)

refq = np.array([0.9802119292455138, 1.9604238584910276, 2.9406357877365417, 3.9208477169820553, 4.901059646227569,  5.881271575473083, 6.861483504718597,  7.8416954339641105, 8.821907363209625, 9.802119292455139])
refsq = np.array([0.2297343191414038, 0.8399849501390582, 0.0619255618120822, 1.4124131697703013, 1.3214070992167133, 0.1397404851571118, 5.608301488475183,  0.4709777234039486, 1.5522548158327347, 2.3958655324307347])
np.testing.assert_allclose(q, refq, rtol=1e-15, atol=1e-15)
np.testing.assert_allclose(sq, refsq, rtol=1e-15, atol=1e-15)


anavdos = AnaSF2VD(s)

fre, vdos = anavdos.vdos(1, 2)
print(fre, vdos)
reffre = np.array([0., 3.2724923474893675e+14, 6.5449846949787350e+14, 9.8174770424681038e+14, 1.3089969389957470e+15, 1.6362461737446838e+15])
refvdos = np.array([0.278641305052774,  0.4332312820127425, 0.303938901747167, 0.0233662557501104, 0.0163422403825281, 0.0119426140994055])
np.testing.assert_allclose(fre, reffre, rtol=1e-15, atol=1e-15)
np.testing.assert_allclose(vdos, refvdos, rtol=1e-15, atol=1e-15)

saveUnwrappedTrj='unittest.tr'
anavdos.saveTrj(saveUnwrappedTrj)
df = DynamicFactor(saveUnwrappedTrj)
Q=0.1
fre, inco = df.calIncoherent(Q)
print(fre, inco)
reffre = np.array([-1.8124573001479575e+15, -1.5103810834566310e+15, -1.2083048667653050e+15, -9.0622865007397875e+14, -6.0415243338265250e+14, -3.0207621669132625e+14, 0.0000000000000000e+00, 3.0207621669132625e+14,  6.0415243338265250e+14, 9.0622865007397875e+14,  1.2083048667653050e+15,  1.5103810834566310e+15, 1.8124573001479575e+15])
np.testing.assert_allclose(fre, reffre, rtol=1e-15, atol=1e-15)
refinco = np.array([1.9950446310901355e-03, 2.2584828159330872e-03, 2.9730587436343956e-03,
 4.7874363363678389e-03, 1.6834008593541075e-02, 8.2222189872508483e-02,
 1.2167777696640736e+04, 8.2321571698239393e-02, 1.6884152658791472e-02,
 4.7959080898435256e-03, 2.9762363175349906e-03, 2.2598436133325190e-03,
 1.9954258921316406e-03])
np.testing.assert_allclose(refinco, inco, rtol=1e-15, atol=1e-15)

fre, inco = df.calIncoherent(Q, True)
print(inco)
refinco = np.array([6.1993955983831153e-06, 6.7266091401902966e-03, 7.3251727080864804e-01,
 1.8515515346552654e+01, 1.5750676581101516e+02, 5.3831833299953826e+02,
 8.0391654511159766e+02, 5.3786896208096118e+02, 1.5720229911660491e+02,
 1.8445374092775594e+01, 7.2678508609072867e-01, 6.5900536401919370e-03,
 5.6949552730480839e-06])
np.testing.assert_allclose(refinco, inco, rtol=1e-15, atol=1e-15)

fre, co = df.calCoherent(Q)
print(co)
refco = np.array([1.4117563871926776e-03, 1.6265455501273752e-03, 2.1487503142946967e-03,
 3.9490890097329413e-03, 8.1227568974644448e-03, 4.8697856089854115e-02,
 8.4644226135600090e+05, 4.8163802065004646e-02, 1.6552506456601855e-02,
 3.2475146910638528e-03, 2.0230468514142071e-03, 1.5406878014335812e-03,
 1.3894959341432896e-03])
np.testing.assert_allclose(co, refco, rtol=1e-15, atol=1e-15)


# import matplotlib.pyplot as plt
# fre, inco = df.calIncoherent(Q)
# plt.semilogy(fre*units.hbar, inco/inco.max(), label='S(Q, $\omega$)')
# fre, inco = df.calIncoherent(Q, True)
# plt.semilogy(fre*units.hbar, inco/inco.max(), label='S(Q, $\omega$) windowed')
# plt.legend()
#
# plt.figure()
# fre, co = df.calCoherent(Q)
# plt.semilogy(fre*units.hbar, co/co.max(), label='coherent S(Q, $\omega$)')
# plt.legend()
# plt.show()
