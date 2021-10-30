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
  //set the seed for the random generator
  pt::Singleton<pt::SingletonPTRand>::getInstance().setSeed(0);

  //load geometry
  auto &geoman = pt::Singleton<pt::GeoManager>::getInstance();
  geoman.loadFile("../gdml/first_geo.gdml");

  //create navigation manager
  auto &navman = pt::Singleton<pt::NavManager>::getInstance();

  size_t numBeam = 10;
  pt::ProgressMonitor moni("Prompt simulation", numBeam);
  for(size_t i=0;i<numBeam;i++)
  {
    pt::Neutron neutron(0.0253 , {0,0,1}, /*pos*/ {0,0,-400.});
    //! allocate the point in a volume
    navman.locateLogicalVolume(neutron.getPosition());
    while(!navman.exitWorld())
    {
      //! first step of a particle in a volume
      // std::cout << navman.getVolumeName() << " " << neutron.getPosition() << std::endl;
      navman.setupVolumePhysics();

      //! the next while loop, particle should move in the same volume
      while(navman.proprogateInAVolume(neutron, 1))
      {
        if(neutron.isAlive())
          continue;
      }
    }
    moni.OneTaskCompleted();
  }
}
