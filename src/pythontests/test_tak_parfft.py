#!/usr/bin/env python3

import numpy as np
import scipy.fft
from Cinema.Tak.Analysor import parFFT, parFFTs

data = np.random.random((128,128)) + np.random.random((128,128))*1j
out = parFFT(data, numcpu=1)
outscipy = scipy.fft.fft(data)
np.testing.assert_allclose(out, outscipy)


out = parFFTs(data, 0, 4, numcpu=1)
scipydata = data[::4]
outscipy = scipy.fft.fft(scipydata)
np.testing.assert_allclose(out, outscipy)
