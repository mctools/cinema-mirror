#ifndef Prompt_PTGIDIElasticModel_hh
#define Prompt_PTGIDIElasticModel_hh

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

#include <string>
#include <memory>
#include <set>

#include "PromptCore.hh"
#include "PTDiscreteModel.hh"
#include "PTSingleton.hh"
#include "PTCentralData.hh"
#include "PTLauncher.hh"
#include "PTGIDIModel.hh"

namespace MCGIDI
{
  class Protare;
}  
#include "PTNCrystalScat.hh"

namespace Prompt {


  class GIDIElasticModel: public GIDIModel{
  public:
    GIDIElasticModel(const std::string &name, std::shared_ptr<MCGIDI::Protare> mcprotare,
                    double temperature, double bias=1.0, double frac=1.0,
                    double lowerlimt = 0., double upperlimt = std::numeric_limits<double>::max());
    virtual ~GIDIElasticModel() = default;

    virtual void generate(double ekin, const Vector &dir, double &final_ekin, Vector &final_dir) const override;

  protected:
    std::shared_ptr<NCrystalScat> m_ncscatt;
  };
}

#endif
