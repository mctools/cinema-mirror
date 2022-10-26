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

#include "PTScorerRotatingObj.hh"

Prompt::ScorerRotatingObj::ScorerRotatingObj(const std::string &name, const Vector &dir, const Vector &point,
  double rotFreq, Scorer::ScorerType type)
:Scorer1D("ScorerRotatingObj_"+name, type,
  std::make_unique<Hist1D>("ScorerRotatingObj_"+name, 0, 1, 100)),
  m_dir(dir), m_point(point), m_angularfreq(2*M_PI*rotFreq)

{
  //fixme use m_dir.normalise() to make sure the accuracy of the conversion
  if(!floateq(m_dir.mag(),1., 1e-5, 1e-5))
    PROMPT_THROW(BadInput, "direction must be a unit vector");
}


Prompt::ScorerRotatingObj::~ScorerRotatingObj() {}

void Prompt::ScorerRotatingObj::score(Prompt::Particle &particle)
{
  std::cout << "RotatingObj " << particle.getPosition() << " " << particle.getStep() << std::endl;
}

Prompt::Vector Prompt::ScorerRotatingObj::getLinearVelocity(const Vector &pos)
{
  Vector B = pos - m_point;
  return (B-m_dir*(m_dir.dot(B)))*m_angularfreq;
}
