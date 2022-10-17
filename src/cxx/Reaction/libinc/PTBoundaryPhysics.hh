#ifndef Prompt_BoundaryPhysics_hh
#define Prompt_BoundaryPhysics_hh

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
#include "PTParticle.hh"
#include "PTRandCanonical.hh"
#include "PTPhysicsModel.hh"

namespace Prompt {
  class BoundaryPhysics : public PhysicsBase  {
  public:
    // by default the physics is applicable for neutron (2112)
    BoundaryPhysics(unsigned pgd=2112): PhysicsBase("NeutronDiskChopper", pgd, std::numeric_limits<double>::min(), std::numeric_limits<double>::max()) {};
    virtual ~BoundaryPhysics() = default;
    virtual void sampleFinalState(Particle &particle, Vector ref = {1,0,0}) const = 0;

  };

}

#endif
