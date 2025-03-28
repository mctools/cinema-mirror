#ifndef Prompt_DiscreteModel_hh
#define Prompt_DiscreteModel_hh

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

#include <string>

#include "PromptCore.hh"
#include "PTPhysicsModel.hh"
#include <memory>

namespace Prompt {

  class DiscreteModel  : public PhysicsModel {
  public:
    DiscreteModel(const std::string &name, double bias=1.0);
    DiscreteModel(const std::string &name, unsigned gdp, double emin, double emax, double bias=1.0);
    virtual ~DiscreteModel();
    double getBias() const
    { return m_bias; }

  protected:
    mutable double m_bias;
  };

}

#endif
