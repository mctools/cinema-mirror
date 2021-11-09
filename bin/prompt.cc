#include <iostream>
#include <getopt.h> //long_options
#include <memory>
#include "PTMath.hh"
#include "PTLauncher.hh"
#include "PTMaxwellianGun.hh"
#include "PTNeutron.hh"


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

  auto gun = std::make_shared<pt::MaxwellianGun>(pt::Neutron(), 300,  std::array<double, 6> {1, 1, -12000, 1, 1, 0});
  auto &l = pt::Singleton<pt::Launcher>::getInstance();
  l.setSeed(seed);
  l.setGun(gun);
  l.loadGeometry(geofile);
  l.go(numParticle, printPrecent);

  return 0;
}
