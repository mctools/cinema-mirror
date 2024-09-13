#!/usr/bin/env python3

import numpy as np

const_neutron_mass = 1.674e-27 #kg
const_neutron_mass_amu = 1.00866491588
const_c  = 299792458 # (m/s)/c
const_dalton2eVc2 =  931494095.17
const_neutron_mass_evc2 = const_neutron_mass_amu * const_dalton2eVc2  #[ eV/(Aa/s)^2 ]

delta_t = 6 * 1e-6 #s

def uniform_ste2timewin(ste):
    return np.sqrt(12 * ste**2)

delta_t = uniform_ste2timewin(delta_t)
theta_degree = 5
theta=np.deg2rad(theta_degree) #rad
radius=17.5 #mm
length_slit = theta * radius #mm

angle_velocity = theta / delta_t 
oneRoundtime = 2*np.pi/angle_velocity
frequency = 1 / oneRoundtime

distance_source_chopper = 10000 + 25000 #mm
distance_source_chopper *= 1e-3 # convert to m

def get_energy(velocity):
    return 0.5 * const_neutron_mass_evc2 / const_c**2  * (velocity)**2

def get_period(angle_velocity):
    return 2 * np.pi / angle_velocity  #s

def get_time_window(theta_rad, angle_velocity, order, phase = 0):
    angle_window = np.array([-theta_rad/2, theta_rad/2]) + phase
    period = get_period(angle_velocity)
    time_window = angle_window / angle_velocity + period * order
    return time_window

def get_gun_energy_window(time_window, distance2Chopper):
    velocity_window = distance2Chopper / time_window
    energy_window = get_energy(velocity_window)
    print(f'Time window: [{time_window[0]}, {time_window[1]}]; Unit: s')
    print(f'Energy window: [{energy_window[1]}, {energy_window[0]}; Unit: eV\n')

def time_flight2distance(time_window, distance2Chopper, distance2destination):
    print(f'Neutron from chopper fly to destination time: {time_window * distance2destination / distance2Chopper}')
    

if __name__ == '__main__':
    for i in np.arange(6):
        print(f'theta_degree: {theta_degree}, frequency: {frequency}')
        print(f'Order: {i}' )
        tw = get_time_window(theta, angle_velocity, i, phase=np.deg2rad(110))
        # time_flight2distance(tw, distance_source_chopper, (10000+ 25000)*1e-3)
        get_gun_energy_window(tw, distance_source_chopper)