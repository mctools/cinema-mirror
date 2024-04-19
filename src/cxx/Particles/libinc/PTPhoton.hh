#ifndef Prompt_Photon_hh
#define Prompt_Photon_hh

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

#include "PTParticle.hh"
#include "PTMath.hh"
//! fixme: support Gamma (22) as well.
namespace Prompt {
  class Photon : public Particle {
  public:
    Photon();
    Photon(double ekin, const Vector& dir, const Vector& pos);
    virtual ~Photon(){};
  };
}

inline Prompt::Photon::Photon()
:Particle(const_photon_pgd)
{
}

inline Prompt::Photon::Photon(double ekin, const Vector& dir, const Vector& pos)
:Particle(ekin, dir, pos, const_photon_pgd)
{
}

#endif
