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

#include "PTScororPSD.hh"

Prompt::ScororPSD::ScororPSD(const std::string &name, double xmin, double xmax, unsigned nxbins, double ymin, double ymax, unsigned nybins)
:Scoror2D("ScororPSD_"+name, Scoror::SURFACE, std::make_unique<Hist2D>(xmin, xmax, nxbins, ymin, ymax, nybins))
{}

Prompt::ScororPSD::~ScororPSD() {}

void Prompt::ScororPSD::scoreLocal(const Vector &vec, double w)
{
  m_hist->fill(vec.x(), vec.y(), w);
}

void Prompt::ScororPSD::score(Prompt::Particle &particle)
{
  PROMPT_THROW2(BadInput, m_name << " does not support score()");
}

void Prompt::ScororPSD::score(Prompt::Particle &particle, const DeltaParticle &dltpar)
{
  PROMPT_THROW2(BadInput, m_name << " does not support score()");
}
