#include "../doctest.h"
#include <iostream>

#include "PTMirrorPhysics.hh"
#include "PromptCore.hh"

namespace pt = Prompt;

TEST_CASE("Mirror physics")
{
  auto mirr = pt::MirrorPhyiscs( "cfg" );
  pt::Vector dir{0, 0.9, 0.01}, nor{0, 0., 1};
  dir.normalise();
  double ekin(0.0253), eout(0), wscale(0.);

  mirr.generate(ekin, dir, eout, nor, wscale);
  printf("%.16g %.16g %.16g, %.16e\n", nor.x(), nor.y(), nor.z(), wscale);
  CHECK(pt::floateq(nor.x(), 0));
  CHECK(pt::floateq(nor.y(), 0.9999382773199424));
  CHECK(pt::floateq(nor.z(), -0.01111042530355492));


}
