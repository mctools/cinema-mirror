#include "../doctest.h"

#include <iostream>
#include <memory>
#include "PTGeoManager.hh"
#include "PTNavManager.hh"
#include "PTMath.hh"
#include "PTNeutron.hh"

namespace pt = Prompt;

TEST_CASE("GeoManager")
{
  pt::Singleton<pt::SingletonPTRand>::getInstance().setSeed(0);

  auto &geoman = pt::Singleton<pt::GeoManager>::getInstance();
  geoman.loadFile("../gdml/first_geo.gdml");

  auto &navman = pt::Singleton<pt::NavManager>::getInstance();

  for(unsigned i=0;i<10000;i++)
  {
    pt::Neutron neutron( 0.0253 , {0,0,1}, /*pos*/ {0,0,-400.});
    navman.locateLogicalVolume(neutron.getPosition());
    while(!navman.exitWorld())
    {
      //first step of a particle in a volume
      navman.setupVolumePhysics();
      while(navman.proprogateInAVolume(neutron, 0))
        continue;
    }
  }
}
