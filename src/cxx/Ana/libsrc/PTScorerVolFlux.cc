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

#include "PTScorerVolFlux.hh"

Prompt::ScorerVolFlux::ScorerVolFlux(const std::string &name, double xmin, double xmax, unsigned nxbins, bool linear, double volme)
:Scorer1D("ScorerVolFlux_"+ name, Scorer::PROPAGATE, std::make_unique<Hist1D>(xmin, xmax, nxbins, linear)), m_iVol(1./volme)
{ }

Prompt::ScorerVolFlux::~ScorerVolFlux() {}

void Prompt::ScorerVolFlux::score(Particle &particle)
{
  if(particle.getWeightFactor()!=1.)
    PROMPT_THROW(LogicError, "ScorerVolFlux is incorrect in the cross section biasing model. The D value for the material within the solid of insterest should be unity");
  m_hist->fill(particle.getEKin()+particle.getEnergyChange(), m_iVol*particle.getStep());
}
