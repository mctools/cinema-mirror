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
#include "PTMath.hh"
#include "PTCompoundModel.hh"

TEST_CASE("CompoundModel")
{

  auto collection = Prompt::CompoundModel(2112) ;
  collection.addPhysicsModel("Al_sg225.ncmat;dcutoff=0.5;temp=25C");

  double xs(0.);
  xs = collection.totalCrossSection(1., {0,0,0} );
  Prompt::Vector out;
  double final;
  std::cout << xs << std::endl;
  printf("%.15f\n", xs);

  CHECK(Prompt::floateq(1.378536096609809*Prompt::Unit::barn, xs ));


  collection.generate(1., {1,0,0}, final, out);
  std::cout << final << " " << out << std::endl;

  collection.generate(1., {1,0,0}, final, out);
  std::cout << final << " " << out << std::endl;

  collection.generate(1., {1,0,0}, final, out);
  std::cout << final << " " << out << std::endl;
}
