#include <iostream>
#include <getopt.h> //long_options
#include <memory>
#include "PTMath.hh"
#include "PTLauncher.hh"
#include "PTMaxwellianGun.hh"
#include "PTNeutron.hh"
#include "PTMeshHelper.hh"


namespace pt = Prompt;

int main(int argc, char *argv[])
{
  //default parameters
  size_t numParticle = 100;
  unsigned seed = 6402;
  bool vis = false;
  double printPrecent = 0.1;
  std::string geofile("../gdml/mpi_detector.gdml");
  bool printTrj = false;

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

  while((opt_char = getopt_long(argc, argv, "n:s:g:vth",
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
      case 't':
      {
        printTrj=true;
        break;
      }
      case 'g':
      {
        geofile=std::string(optarg);
        printf("geometry file is set to %s\n", optarg);
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
        return 0;
      }
    }
  }

  auto &l = pt::Singleton<pt::Launcher>::getInstance();
  l.setSeed(seed);
  auto gun = std::make_shared<pt::MaxwellianGun>(pt::Neutron(), 300,  std::array<double, 6> {600, 600, -30000, 10, 10, -1000});
  l.setGun(gun);
  l.loadGeometry(geofile);
  // pt_printMesh();
  l.go(numParticle, printPrecent, printTrj);

  if(printTrj)
  {
    auto &trj = l.getTrajectory();
    for(const auto &v : trj )
    {
      std::cout << v << std::endl;
    }
  }

  return 0;
}
