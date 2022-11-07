#ifndef Prompt_PhysicsFactory_hh
#define Prompt_PhysicsFactory_hh

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

#include "PromptCore.hh"
#include "PTSingleton.hh"
#include "PTBoundaryPhysics.hh"
#include "PTCompoundModel.hh"

namespace Prompt {

  class PhysicsFactory  {
  public:
    enum class PhysicsType {
      BOUNDARY_PHYSICS,
      NC_SCATTER,
      NC_ABSORB,
      NC_SCATTER_ABSORB,
      ENDF_SCATTER,
      ENDF_ABSORB
    };
  public:
    bool pureNCrystalCfg(const std::string &cfgstr);
    double calNumDensity(const std::string &nccfgstr); // ncrystal cfg string

    std::shared_ptr<BoundaryPhysics> createBoundaryPhysics(const std::string &cfgstr);
    std::unique_ptr<CompoundModel> createBulkPhysics(const std::string &cfgstr);
    PhysicsType checkPhysicsType(const std::string &cfgstr) const;

  private:
    friend class Singleton<PhysicsFactory>;
    PhysicsFactory() = default;
    ~PhysicsFactory() = default;
  };
}

#endif
