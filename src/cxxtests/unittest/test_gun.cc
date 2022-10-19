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

#include "PTMaxwellianGun.hh"
#include "PTNeutron.hh"
#include "PTHist1D.hh"

namespace pt = Prompt;

TEST_CASE("test_maxwellgun")
{
  auto hist = std::make_unique<pt::Hist1D>("test_maxwellgun", 0.0001, 0.3 , 100, false);
  auto gun = pt::MaxwellianGun(pt::Neutron(), 300, {1,1,-1400,1,1,0});
  for(unsigned i=0;i<10;i++)
  {
    auto p = gun.generate();
    std::cout << "event id " << p.getEventID()
    << " " << p.getPosition()
    << " " << p.getDirection()<< std::endl;
    hist->fill(p.getEKin());
  }
  std::cout << "integral " << hist->getIntegral() << std::endl;
  hist->save("test_maxwellgun");

}
