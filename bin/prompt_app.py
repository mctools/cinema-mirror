#!/usr/bin/env python3

from prompt import Launcher, Visualiser
import matplotlib.pyplot as plt
import numpy as np

l = Launcher()
l.loadGeometry("../gdml/first_geo.gdml");

v = Visualiser()
v.show()
