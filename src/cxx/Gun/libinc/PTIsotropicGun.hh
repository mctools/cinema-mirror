#ifndef Prompt_IsotropicGun_hh
#define Prompt_IsotropicGun_hh

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
#include "PTParticle.hh"
#include "PTVector.hh"
#include "PTSimpleThermalGun.hh"

namespace Prompt {
  class IsotropicGun : public SimpleThermalGun {
  public:
    IsotropicGun(const Particle &aParticle, double ekin, const Vector &pos, const Vector &dir )
    : SimpleThermalGun(aParticle, ekin, pos, dir), m_pos(pos) {};
    virtual ~IsotropicGun() {};
    virtual void samplePosDir(Vector &pos, Vector &dir) override
    { 
        pos = m_pos;
        double theta = M_PI*m_rng.generate();
        double fai = 2*M_PI*m_rng.generate();

        double sin_theta = sin(theta);
        double dir_x = sin_theta*cos(fai);
        double dir_y = sin_theta*sin(fai);
        double dir_z = cos(theta);
        dir = Vector(dir_x, dir_y, dir_z);
    }
  private:
    Vector m_pos;
  };
}


#endif