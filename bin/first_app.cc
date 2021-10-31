#include <iostream>
#include <memory>
#include "PTGeoManager.hh"
#include "PTNavManager.hh"
#include "PTMath.hh"
#include "PTNeutron.hh"
#include "PTProgressMonitor.hh"
#include "PTMaxwellianGun.hh"

namespace pt = Prompt;

int main(int argc, char *argv[])
{
  //set the seed for the random generator
  auto &rng = pt::Singleton<pt::SingletonPTRand>::getInstance();
  rng.setSeed(0);

  //load geometry
  auto &geoman = pt::Singleton<pt::GeoManager>::getInstance();
  geoman.loadFile("../gdml/first_geo.gdml");

  //create navigation manager
  auto &navman = pt::Singleton<pt::NavManager>::getInstance();


  size_t numBeam = atoi(argv[1]);
  pt::ProgressMonitor moni("Prompt simulation", numBeam);

  auto gun = pt::MaxwellianGun(pt::Neutron(), 300, {20, 20, -12000, 4, 4,0});

  for(size_t i=0;i<numBeam;i++)
  {
    //double ekin, const Vector& dir, const Vector& pos
    auto neutron = gun.generate();
    // double sampleHalfSize = 2.;
    // pt::Neutron neutron(0.3-rng.generate()*0.25, {rng.generate()*1e-5,rng.generate()*1e-5,1.},
    //   {rng.generate()*sampleHalfSize*2-sampleHalfSize,rng.generate()*sampleHalfSize*2-sampleHalfSize, -12000.*pt::Unit::mm});

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
  return 0;
}
