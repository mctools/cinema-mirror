#!/usr/bin/env python3

from Cinema.Interface import CinemaArray
import numpy as np

a = CinemaArray.from_counts([100,1000,10000])
b = CinemaArray.from_sdev([100,1000,10000], [10, np.sqrt(1000), 100])
aplusb = CinemaArray.from_sdev([200, 2000, 20000], np.sqrt([200, 2000, 20000])) 

np.testing.assert_array_equal((a+b).mean, aplusb.mean)
np.testing.assert_array_equal((a+b).sdev, aplusb.sdev)

