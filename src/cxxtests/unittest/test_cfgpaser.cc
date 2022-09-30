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

#include "PTCfgParser.hh"

namespace pt = Prompt;
#include <typeinfo>
#include <iostream>
#include <vector>

TEST_CASE("CfgParser")
{
  pt::CfgParser ps;
  std::cout << ps.getTypeName(typeid(pt::CfgParser)) << std::endl;
  auto cfg = ps.getScorerCfg("Scorer=(NeutronSq); name=str(SofQ);sample_position=vec(0,0,1);beam_direction=vec(0,0,1);src_sample_dist=double(30000);ScorerType=str(ENTRY);linear=bool(true)");
  cfg.print();
}
