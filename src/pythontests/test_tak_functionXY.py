#!/usr/bin/env python3

import unittest
import numpy as np
import numpy.testing as npt

class MainTest(unittest.TestCase):
    def setUp(self):
        pass

    def dummy_test(self):
        self.assertAlmostEqual(1.+1e-16, 1. , 14)

    def test_functionXY(self):
        from Cinema.Tak.FunctionXY import FunctionXY
        from Cinema.Tak.helper import getOmegaFromTime, takconv
        # import matplotlib.pyplot as plt
        dt=0.2
        data1=FunctionXY(np.linspace(-1, -1+9*dt, 10), np.ones(10)*1.5)
        data2=FunctionXY(np.linspace(-2, -2+19*dt, 20), np.ones(20))
        # plt.plot(data1.x, data1.y, 'o', label='d1')
        # plt.plot(data2.x, data2.y,'s', label='d2')

        ############ accumulate #######################
        data1.accumulate(data2)

        dataSum=np.ones(20)
        dataSum[5:15]=2.5
        # plt.plot(data1.x, data1.y, 'd', label='d1+d2')
        # plt.show()

        npt.assert_array_almost_equal(data1.y, dataSum, decimal =  15)

        #==============cutoff===================
        cutoffValue=1.3
        idx1=np.where(data1.x>=-cutoffValue)[0][0]
        idx2=np.where(data1.x<=cutoffValue)[0][-1]
        #tmpx = data1.x[idx1:idx2]
        tmpy = data1.y[idx1:(idx2+1)]
        data1.crop(-cutoffValue, cutoffValue)
        npt.assert_array_almost_equal(data1.y, tmpy, decimal =  15)
        ############# convolve ##########################
        #plt.figure()
        res = takconv(data1, data2)
        #plt.plot(res.x, res.f, 'o')
        y = np.convolve(data1.y, data2.y)
        #plt.plot(res.x, y*dt)
        npt.assert_array_almost_equal(res.y,  y*dt, decimal =  15)

        ########distortion and recovery#####################
        #plt.figure()
        dw, omega = getOmegaFromTime(30, 0.1)
        distortf=2.5*2
        asymExp=2.
        y=np.exp(-omega*omega/30)*np.exp(omega*asymExp)+1j*0
        data3=FunctionXY(omega, y, distortFact = distortf, asymExponent=asymExp*2)
        original = data3.y
        # plt.figure()
        # plt.semilogy(data3.x, data3.y,'o',alpha=0.3)
        data3.flipNeg2Pos()
        flip=data3.y
        # plt.semilogy(data3.x, data3.y,'s',alpha=0.3)
        # plt.show()
        npt.assert_allclose(original, flip[:-1], rtol=1e-12)
        # plt.semilogy(data3.x, data3.y, '>', label='flipNeg2Pos')
        data3.distort()
        # plt.semilogy(data3.x, data3.y, '<', label='distort')

        dist=data3.y
        data3.flipNeg2Pos()
        npt.assert_allclose(data3.y, dist, rtol=1e-12)
        # plt.semilogy(data3.x, data3.y, 'D', label='flipNeg2Pos')

        data3.restort()
        # plt.semilogy(data3.x, data3.y, 'o', label='recover')
        npt.assert_allclose(data3.y[:-1], original, rtol=1e-12, atol=1e-12)
        # plt.legend()
        # plt.show()

if __name__ == '__main__':
    unittest.main()
