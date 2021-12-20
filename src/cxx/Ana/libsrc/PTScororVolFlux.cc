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

#include "PTScororVolFlux.hh"

Prompt::ScororVolFlux::ScororVolFlux(const std::string &name, double xmin, double xmax, unsigned nxbins, bool linear, double volme)
:Scoror1D("ScororVolFlux_"+ name, Scoror::PROPAGATE, std::make_unique<Hist1D>(xmin, xmax, nxbins, linear)), m_iVol(1./volme)
{ }

Prompt::ScororVolFlux::~ScororVolFlux() {}


void Prompt::ScororVolFlux::scoreLocal(const Vector &vec, double w)
{
  PROMPT_THROW2(BadInput, m_name << " does not support scoreLocal()");
}

void Prompt::ScororVolFlux::score(Particle &particle)
{
  PROMPT_THROW2(BadInput, m_name << " does not support score(Particle &particle)");
}

void Prompt::ScororVolFlux::score(Particle &particle, const DeltaParticle &dltpar)
{
  //w=m_iVol*dltpar.dlt_pos.mag()*particle.getWeight()
  m_hist->fill(particle.getEKin()-dltpar.dlt_ekin, m_iVol*dltpar.dlt_pos.mag());
}
