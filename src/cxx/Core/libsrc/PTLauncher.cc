#include "PTLauncher.hh"

#include "PTGeoManager.hh"
#include "PTNavManager.hh"
#include "PTMath.hh"
#include "PTParticle.hh"
#include "PTProgressMonitor.hh"
#include "PTSimpleThermalGun.hh"
#include "PTNeutron.hh"

Prompt::Launcher::Launcher()
{

}


Prompt::Launcher::~Launcher()
{
  printLogo();
}

void Prompt::Launcher::loadGeometry(const std::string &geofile)
{
  //load geometry
  auto &geoman = Singleton<GeoManager>::getInstance();
  geoman.loadFile(geofile.c_str());
}


void Prompt::Launcher::go(uint64_t numParticle, double printPrecent, bool recordTrj)
{
  //set the seed for the random generator
  auto &rng = Singleton<SingletonPTRand>::getInstance();

  //create navigation manager
  auto &navman = Singleton<NavManager>::getInstance();

  if(!m_gun.use_count())
  {
    std::cout << "PrimaryGun is not set, fallback to the neutron SimpleThermalGun\n";
    m_gun = std::make_shared<SimpleThermalGun>(Neutron());
  }

  DeltaParticle dltpar;

  ProgressMonitor moni("Prompt simulation", numParticle, printPrecent);
  for(size_t i=0;i<numParticle;i++)
  {
    //double ekin, const Vector& dir, const Vector& pos
    auto particle = m_gun->generate();
    dltpar.setLastParticle(particle);

    if(recordTrj)
    {
      std::vector<Vector> tmp;
      tmp.reserve(m_trajectory.size());
      m_trajectory.swap(tmp);
    }

    //! allocate the point in a volume
    navman.locateLogicalVolume(particle.getPosition());
    while(!navman.exitWorld() && particle.isAlive())
    {
      if(recordTrj)
      {
        m_trajectory.push_back(particle.getPosition());
      }

      //! first step of a particle in a volume
      // std::cout << navman.getVolumeName() << " " << particle.getPosition() << std::endl;
      navman.setupVolumePhysics();
      navman.scoreEntry(particle);

      //! the next while loop, particle should move in the same volume
      while(navman.proprogateInAVolume(particle, 0) )
      {
        if(navman.hasPropagateScoror())
        {
          dltpar.calcDeltaParticle(particle);
          navman.scorePropagate(particle, dltpar);
        }
        if(recordTrj)
          m_trajectory.push_back(particle.getPosition());
      }
      navman.scoreExit(particle);
    }
    moni.OneTaskCompleted();
  }
}
