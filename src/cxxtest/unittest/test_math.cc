#include "../doctest.h"
#include <iostream>

#include "PTNeutron.hh"
#include "PTMath.hh"
#include "PTVector.hh"

namespace pt = Prompt;

TEST_CASE("Math")
{
  auto n = pt::Neutron( 100, pt::Vector(1,0,0), pt::Vector(0,0,0) );
  printf("%.16g\n", n.calcSpeed());
}
