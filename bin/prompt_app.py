#!/usr/bin/env python3

from prompt import Launcher, Visualiser
import matplotlib.pyplot as plt
import numpy as np

l = Launcher()
l.setSeed(100)
l.loadGeometry("../gdml/mpi_detector.gdml");
# l.loadGeometry("../gdml/first_geo.gdml");
v = Visualiser(['Tube300', 'Tube500', '300_L', '500_L'])

for i in range(10):
    print(f'trajectory size {l.getTrajSize()}')
    l.go(1, recordTrj=True)
    print(f'trajectory size {l.getTrajSize()}')
    trj = l.getTrajectory()
    v.addLine(trj)
    print(trj)


v.show()
