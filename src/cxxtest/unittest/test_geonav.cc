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
  auto &geoman = pt::Singleton<pt::GeoManager>::getInstance();
  geoman.loadFile("../gdml/first_geo.gdml");

  auto &navman = pt::Singleton<pt::NavManager>::getInstance();

  pt::Neutron neutron(1 , {1,0,0}, {0,0,0});
  while(navman.proprogate(neutron))
    continue;
}
