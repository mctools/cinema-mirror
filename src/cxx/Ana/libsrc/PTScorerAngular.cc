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

#include "PTScorerAngular.hh"

Prompt::ScorerAngular::ScorerAngular(const std::string &name, const Vector &samplePos, const Vector &refDir,
      double sourceSampleDist, double mu_min, double mu_max, unsigned numbin, ScorerType stype, bool linear)
:Scorer1D("ScorerAngular_" + name, stype, std::make_unique<Hist1D>("ScorerAngular_" + name, 
mu_min==-1?mu_min-1e-14:mu_min, 
mu_max==1?mu_max+1e-14:mu_max, 
numbin, linear)), m_samplePos(samplePos), m_refDir(refDir.unit()), 
m_sourceSampleDist(sourceSampleDist)
{
  // if(mu_max>1 || mu_min<-1 || mu_min>=mu_max)
  //   PROMPT_THROW2(BadInput, "angular range should be within 0 to 180 degrees" )
}

Prompt::ScorerAngular::~ScorerAngular()
{
}


void Prompt::ScorerAngular::score(Prompt::Particle &particle)
{
  if((m_refDir.angleCos(particle.getDirection()))>1)
  {
    std::cout << "wrong mu " << m_refDir.angleCos(particle.getDirection()) << std::endl;

  }
  m_hist->fill(m_refDir.angleCos(particle.getDirection()), particle.getWeight());  
  // double angle_cos = (m_samplePos-particle.getPosition()).angleCos(m_refDir);
  // m_hist->fill(180-std::acos(angle_cos)*const_rad2deg, particle.getWeight());
}
