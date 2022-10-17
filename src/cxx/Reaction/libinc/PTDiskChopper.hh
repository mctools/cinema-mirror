#ifndef Prompt_DiskChopper_hh
#define Prompt_DiskChopper_hh

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
#include "PTBoundaryPhysics.hh"

namespace Prompt {

  class DiskChopper  : public BoundaryPhysics
  {
    public:
      enum class ActiveFace {XY, XZ, YZ};

    public:
      DiskChopper(const Vector &centre, double radius, double theta0, double h, double phase, double freq);
      virtual ~DiskChopper() = default;
      virtual void sampleFinalState(Particle &particle) const override;

    private:
      Vector m_centre;
      double m_theta0, m_h, m_phase, m_freq;
  };

}

#endif
