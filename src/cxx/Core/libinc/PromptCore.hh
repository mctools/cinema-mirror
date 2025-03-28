#ifndef Prompt_Core_hh
#define Prompt_Core_hh

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

#include <cmath>
#include <vector>
#include <iostream>
#include <limits>
#include <memory>
#include <map>
#include <cassert>
#include <string>

#include "PTException.hh"
#include "PTUnitSystem.hh"
#include "PTVector.hh"
#include "PTMath.hh"

namespace Prompt
{

  constexpr double ENERGYTOKEN_ABSORB = -1.;
  constexpr double ENERGYTOKEN_BIAS = -2.;
  constexpr double ENERGYTOKEN_SCORE = -3.;

  static const std::string PTVersion = "v1.0.0";

  void printLogo();
  void printLogo2();
}
#endif
