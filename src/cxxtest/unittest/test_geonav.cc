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

  auto gen = std::make_shared<std::mt19937_64> (6402);
  auto r = pt::RandCanonical<std::mt19937_64>(gen);


  size_t numBeam = 100000000;
  pt::ProgressMonitor moni("Prompt simulation", numBeam);
  for(size_t i=0;i<numBeam;i++)
  {
    //double ekin, const Vector& dir, const Vector& pos
    pt::Neutron neutron(0.05 , {r.generate()*1e-3,r.generate()*1e-3,1.}, {r.generate()*3,r.generate()*3,-4000.*pt::Unit::mm});
    //! allocate the point in a volume
    navman.locateLogicalVolume(neutron.getPosition());
    while(!navman.exitWorld() && neutron.isAlive())
    {
      //! first step of a particle in a volume
      // std::cout << navman.getVolumeName() << " " << neutron.getPosition() << std::endl;
      navman.setupVolumePhysics();

      //! the next while loop, particle should move in the same volume
      while(navman.proprogateInAVolume(neutron, 0))
      {
        if(neutron.isAlive())
          continue;
      }
    }
    moni.OneTaskCompleted();
  }
}
