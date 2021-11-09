#include "PTLauncher.hh"

#include "PTGeoManager.hh"
#include "PTNavManager.hh"
#include "PTMath.hh"
#include "PTParticle.hh"
#include "PTProgressMonitor.hh"


Prompt::Launcher::Launcher()
{

}


Prompt::Launcher::~Launcher()
{

}

void Prompt::Launcher::loadGeometry(const std::string &geofile)
{
  //load geometry
  auto &geoman = Singleton<GeoManager>::getInstance();
  geoman.loadFile(geofile.c_str());
}


void Prompt::Launcher::go(uint64_t numParticle, double printPrecent)
{
  //set the seed for the random generator
  auto &rng = Singleton<SingletonPTRand>::getInstance();

  //create navigation manager
  auto &navman = Singleton<NavManager>::getInstance();

  printLogo();
  // printLogo2();

  ProgressMonitor moni("Prompt simulation", numParticle, printPrecent);
  for(size_t i=0;i<numParticle;i++)
  {
    //double ekin, const Vector& dir, const Vector& pos
    auto particle = m_gun->generate();
    // auto particle = Neutron(0.1, {0.,0.,1.}, {0,0,-12000.});

    //! allocate the point in a volume
    navman.locateLogicalVolume(particle.getPosition());
    while(!navman.exitWorld() && particle.isAlive())
    {
      //! first step of a particle in a volume
      // std::cout << navman.getVolumeName() << " " << particle.getPosition() << std::endl;
      navman.setupVolumePhysics();
      navman.scoreEntry(particle);

      //! the next while loop, particle should move in the same volume
      while(navman.proprogateInAVolume(particle, 0))
      {
        navman.scorePropagate(particle);
        if(particle.isAlive())
          continue;
      }
      navman.scoreExit(particle);
    }
    moni.OneTaskCompleted();
  }

}
