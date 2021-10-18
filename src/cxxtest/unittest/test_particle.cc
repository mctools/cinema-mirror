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
  double pn_mass_ratio=pt::const_neutron_mass_amu/pt::const_proton_mass_amu;

  auto n = pt::Neutron(0., pt::Vector(1,0,0), pt::Vector(0,0,0) );
  auto p = pt::Proton(0., pt::Vector(1,0,0), pt::Vector(0,0,0) );

  for(auto en : en_vec)
  {
    n.changeEKinTo(en);
    p.changeEKinTo(en);
    printf("ekin %g, speed n %g, p %g\n\n", en, n.calcSpeed(), p.calcSpeed());
    double speedRatio = p.calcSpeed()/n.calcSpeed();
    CHECK(pt::floateq(speedRatio*speedRatio, pn_mass_ratio));
  }

  n.changeEKinTo(pt::const_ekin_2200m_s);
  printf("referce energy at 2200m/s %g \n\n", pt::const_ekin_2200m_s);
  CHECK(pt::floateq(n.calcSpeed(), 2200*pt::Unit::m/pt::Unit::s));

}
