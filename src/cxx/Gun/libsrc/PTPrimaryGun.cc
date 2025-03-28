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

#include "PTPrimaryGun.hh"


std::unique_ptr<Prompt::Particle> Prompt::PrimaryGun::generate()
{
  sampleEnergy(m_ekin);
  m_ekin0=m_ekin;
  samplePosDir(m_pos, m_dir);
  m_eventid++;
  m_weight = getParticleWeight();
  m_alive=true;
  m_time = getTime();
  auto p = std::make_unique<Particle>();
  *p.get() = *this;
  // std::cout << *this << std::endl;
  return std::move(p);
}
