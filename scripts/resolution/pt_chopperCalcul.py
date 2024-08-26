#!/usr/bin/env python3

import numpy as np

delta_t = 38 * 1e-6 #38us

theta_degree = 10
theta=np.deg2rad(theta_degree) #rad
radius=17.5 #mm
length_slit = theta * radius #mm

angle_velocity = theta / delta_t 
oneRoundtime = 2*np.pi/angle_velocity
frequency = 1 / oneRoundtime

print(f'theta_degree: {theta_degree}, frequency: {frequency}')