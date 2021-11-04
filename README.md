# Prompt
Probroblitiy-Conserved Cross Section Biasing Monte Carlo Particle Transport System

Before compiling Prompt, please install the vecgeom package (https://gitlab.cern.ch/VecGeom/VecGeom), which is the only dependence of this Monte Carlo system at the moment. 

Compilation
-----------------------
cd $PATH_OF_PROMPT
mkdir build && cd build
cmake ..
make -j


Version history
----------------------
v beta.0.1, 14 Oct 2021
- Initial draft by compiling distributed modules from TAK, PiXiu, and NCrystal
- PTGeoManager class is able to load gmdl files using the Vecgeom package

v beta.0.2, 31 Oct 2021
- PTSingleton for modern template singleton management. 
- Add volume scoror for statistical analysis   
- Add particle gun module for generate primary incident particle
- Smart pointer for material physics models resource management
