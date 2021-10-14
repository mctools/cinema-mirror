#include "../doctest.h"
#include <iostream>

#include "PTNeutron.hh"
#include "PTProton.hh"

#include "PTMath.hh"
#include "PTVector.hh"
#include <vector>
namespace pt = Prompt;

TEST_CASE("Particle")
{
  auto en_vec = pt::logspace(-5, 2, 100);
  double pn_mass_ratio=pt::const_proton_rest_mass/pt::const_neutron_rest_mass;

  auto n = pt::Neutron(0., pt::Vector(1,0,0), pt::Vector(0,0,0) );
  auto p = pt::Proton(0., pt::Vector(1,0,0), pt::Vector(0,0,0) );

  for(auto en : en_vec)
  {
    n.changeEKinTo(en);
    p.changeEKinTo(en);

    double speedRatio = n.calcSpeed()/p.calcSpeed();
    CHECK(pt::floateq(speedRatio*speedRatio, pn_mass_ratio));
  }
}
