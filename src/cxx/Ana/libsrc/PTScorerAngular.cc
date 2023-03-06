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

#include "PTScorerAngular.hh"

Prompt::ScorerAngular::ScorerAngular(const std::string &name, const Vector &samplePos, const Vector &refDir,
      double sourceSampleDist, double angle_min, double angle_max, unsigned numbin, ScorerType stype, bool linear)
:ScorerNeutronSq(name, samplePos, refDir, sourceSampleDist, angle_min, angle_max, numbin, stype, linear )
{
  if(angle_max>180 || angle_min<0 || angle_min>=angle_max)
    PROMPT_THROW2(BadInput, "angular range should be within 0 to 180 degrees" )
}

Prompt::ScorerAngular::~ScorerAngular()
{
}


void Prompt::ScorerAngular::score(Prompt::Particle &particle)
{
    
  double angle_cos = (m_samplePos-particle.getPosition()).angleCos(m_refDir);
  m_hist->fill(std::acos(angle_cos)*const_rad2deg, particle.getWeight());
}
