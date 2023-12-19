////////////////////////////////////////////////////////////////////////////////
//                                                                            //
//  This file is part of Prompt (see https://gitlab.com/xxcai1/Prompt)        //
//                                                                            //
//  Copyright 2021-2024 Prompt developers                                     //
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

#include <chrono>
#include <iostream>

#include "PTLookUpTable.hh"
#include "PTRandCanonical.hh"
#include "PTRandEngine.hh"
#include "PTMath.hh"
namespace pt = Prompt;

#include "../doctest.h"
TEST_CASE("mt19937_64")
{
  double upper=100;
  auto datax = pt::linspace(0,upper,100);
  auto datay = pt::linspace(0,upper,100);

  auto gen = std::make_shared<std::mt19937_64> (6402);
  auto RandCanonical = pt::RandCanonical<std::mt19937_64>(gen);
  auto lut = std::make_unique<pt::LookUpTable>(datax,datay);


  auto start = std::chrono::steady_clock::now();
  uint64_t loop = 1000000;
  double max(0),sum(0);

  for(uint64_t i=0;i<loop;i++)
  {
    double rdn =RandCanonical.generate();
    double x = rdn*upper;
    double y = lut->get(x);
    CHECK(pt::floateq(x,y));
    max=max<y?y:max;
    sum += rdn;
  }
  auto end = std::chrono::steady_clock::now();
  double nanos = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  printf("max %.16f\n",max);
  printf("sum %.16f\n",sum);
  CHECK(pt::floateq(sum,499930.1475065607228316));
  CHECK(pt::floateq(max,99.9999345781730966));

  std::cout << "Elapsed time in milliseconds : "
    << nanos/1e6 << "ms , " << nanos/loop << " ns per iteration"
    << " " << std::endl;
}
