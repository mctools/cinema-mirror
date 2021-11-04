#include <iostream>
#include <memory>
#include "PTGeoManager.hh"
#include "PTNavManager.hh"
#include "PTMath.hh"
#include "PTNeutron.hh"
#include "PTProgressMonitor.hh"
#include "PTMaxwellianGun.hh"
#include <getopt.h> //long_options

namespace pt = Prompt;

int main(int argc, char *argv[])
{

  //default parameters
  size_t numParticle = 100;
  unsigned seed = 6402;
  bool vis = false;
  double printPrecent = 0.1;
  std::string geofile("../gdml/first_geo.gdml");


  int opt_char;
  int option_index;
  static struct option long_options[] = {
    {"particle", required_argument, 0, 'n'},
    {"printPrecent", required_argument, 0, 'p'},
    {"seed", required_argument, 0, 's'},
    {"geometry", required_argument, 0, 'g'},
    {"vis", no_argument, 0, 'v'},
    {"help",        no_argument,       0, 'h'},
    {NULL,          0,                 0, 0}
  };

  while((opt_char = getopt_long(argc, argv, "n:s:vh",
            long_options, &option_index)) != -1) {
  switch (opt_char) {
      case 'n':
      {
        numParticle=atoi(optarg);
        printf("number of particle is set to %lu\n", numParticle);
        break;
      }
      case 's':
      {
        seed=atoi(optarg);
        printf("seed is set to %d\n", seed);
        break;
      }
      case 'p':
      {
        printPrecent=atof(optarg);
        break;
      }
      case 'v':
      {
        vis=false;
        break;
      }
      case 'h':
      {
        printf("error optopt: %c\n", optopt);
        printf("error opterr: %d\n", opterr);
        break;
      }
    }
  }

  //set the seed for the random generator
  auto &rng = pt::Singleton<pt::SingletonPTRand>::getInstance();
  rng.setSeed(seed);

  //load geometry
  auto &geoman = pt::Singleton<pt::GeoManager>::getInstance();
  geoman.loadFile(geofile.c_str());

  //create navigation manager
  auto &navman = pt::Singleton<pt::NavManager>::getInstance();

  auto gun = pt::MaxwellianGun(pt::Neutron(), 300, {1, 1, -12000, 1, 1, 0});

  pt::printLogo();
  // pt::printLogo2();

  pt::ProgressMonitor moni("Prompt simulation", numParticle, printPrecent);
  for(size_t i=0;i<numParticle;i++)
  {
    //double ekin, const Vector& dir, const Vector& pos
    // auto neutron = gun.generate();
    auto neutron = pt::Neutron(0.1, {0.,0.,1.}, {0,0,-12000.});

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
