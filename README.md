# Prompt
Probability-Conserved Cross Section Biasing Monte Carlo Particle Transport System

Before compiling Prompt, please install the vecgeom package (https://gitlab.cern.ch/VecGeom/VecGeom), which is the only dependence of this Monte Carlo system at the moment.

Compilation
-----------------------
cd $PATH_OF_PROMPT &&
mkdir build && cd build &&
cmake .. &&
make -j


Version history
----------------------
v alpha.0.1, 25 March 2022
- Added more analysor in the Ana pacakge
- Created a new class of particle guns for the simulations of CSNS instruments
- A rich collection of python modules for analysing data produced by Prompt simulation

v beta.0.2, 31 Oct 2021
- PTSingleton for modern template singleton management.
- Add volume scorer for statistical analysis   
- Add particle gun module for generate primary incident particle
- Smart pointer for material physics models resource management

v beta.0.1, 14 Oct 2021
- Initial draft by compiling distributed modules from TAK, PiXiu, and NCrystal
- PTGeoManager class is able to load gmdl files using the Vecgeom package
