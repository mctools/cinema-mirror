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

#include "PTRayTracingProcess.hh"

Prompt::RayTracingProcess::RayTracingProcess()
:Prompt::SurfaceProcess("RayTracing") 
, m_activeVol(Singleton<ActiveVolume>::getInstance()) { }

void Prompt::RayTracingProcess::moveToOut(Prompt::Particle &p) const
{
    double dis = m_activeVol.distanceToOut(p.getPosition(), p.getDirection());
    p.moveForward(dis);
}

void Prompt::RayTracingProcess::sampleFinalState(Prompt::Particle &p) const
{
    canSurvive(p) ? moveToOut(p) : p.kill(Particle::KillType::RT_ABSORB);;
}


