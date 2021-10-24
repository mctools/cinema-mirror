#include "../doctest.h"

#include <iostream>
#include <memory>
#include "PTGeoManager.hh"
#include "PTNavManager.hh"
#include "PTMath.hh"
#include "PTNeutron.hh"
#include "PTProgressMonitor.hh"

namespace pt = Prompt;

TEST_CASE("GeoManager")
{
  pt::Singleton<pt::SingletonPTRand>::getInstance().setSeed(0);

  auto &geoman = pt::Singleton<pt::GeoManager>::getInstance();
  geoman.loadFile("../gdml/first_geo.gdml");

  auto &navman = pt::Singleton<pt::NavManager>::getInstance();

  size_t numBeam = 100000;
  pt::ProgressMonitor moni("Prompt simulation", numBeam);

  for(size_t i=0;i<numBeam;i++)
  {
    pt::Neutron neutron( 0.0253 , {0,0,1}, /*pos*/ {0,0,-400.});
    //! allocate the point in a volume
    navman.locateLogicalVolume(neutron.getPosition());
    while(!navman.exitWorld())
    {
      //! first step of a particle in a volume
      navman.setupVolumePhysics();

      //! the next while loop, particle should move in the same volume
      while(navman.proprogateInAVolume(neutron, 0))
        continue;
    }
    moni.OneTaskCompleted();
  }
}
