////////////////////////////////////////////////////////////////////////////////
//                                                                            //
//  This file is part of Prompt (see https://gitlab.com/xxcai1/Prompt)        //
//                                                                            //
//  Copyright 2021-2022 Prompt developers                                     //
//                                                                            //
//  Licensed under the Apache License, Version 2.0 (the "License");           //
//  you may not use this file except in compliance with the License.          //
//  You may obtain a copy of the License at                                   //
//                                                                            //
//      http://www.apache.org/licenses/LICENSE-2.0                            //
//                                                                            //
//  Unless required by applicable law or agreed to in writing, software       //
//  distributed under the License is distributed on an "AS IS" BASIS,         //
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  //
//  See the License for the specific language governing permissions and       //
//  limitations under the License.                                            //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

#include "../doctest.h"
#include <iostream>

#include "PTMirrorPhysics.hh"
#include "PromptCore.hh"
#include "PTNeutron.hh"

namespace pt = Prompt;

TEST_CASE("Mirror physics")
{
  auto mirr = pt::MirrorPhyiscs( 4. );
  pt::Vector dir{0, 0.9, 0.01}, nor{0, 0., 1};
  dir.normalise();
  double ekin(0.0253), eout(0), wscale(0.);
  auto n = Prompt::Neutron();
  n.setEKin(ekin);
  n.setDirection(dir);
  mirr.setReflectionNormal(nor);
  mirr.sampleFinalState(n);
  wscale = mirr.getEventWeight();
  printf("%.16g %.16g %.16g, %.16e\n", nor.x(), nor.y(), nor.z(), wscale);
  const auto &newdir = n.getDirection();
  CHECK(pt::floateq(newdir.x(), 0));
  CHECK(pt::floateq(newdir.y(), 0.9999382773199424));
  CHECK(pt::floateq(newdir.z(), -0.01111042530355492));
  CHECK(pt::floateq(wscale, 6.5415429925975188e-01));                            
}
