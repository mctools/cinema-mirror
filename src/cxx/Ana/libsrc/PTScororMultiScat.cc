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

#include "PTScororMultiScat.hh"

Prompt::ScororMultiScat::ScororMultiScat(const std::string &name, double xmin, double xmax, unsigned nxbins, bool linear)
:Scoror1D("ScororMultiScat_"+ name, Scoror::PROPAGATE, std::make_unique<Hist1D>(xmin, xmax, nxbins, linear)), m_lasteventid(0), counter(0), p_weight(0)
{ }

Prompt::ScororMultiScat::~ScororMultiScat() {}

void Prompt::ScororMultiScat::score(Particle &particle)
{
  if (m_lasteventid==particle.getEventID())
  {
    counter++;
    p_weight=particle.getWeight();
  } 
  else
  {
    if(counter==0)
    {
      m_lasteventid=particle.getEventID();
      counter=1;
      p_weight=particle.getWeight();
    }
    else
    {
      m_hist->fill(counter, p_weight);
      m_lasteventid=particle.getEventID();
      counter=1;
      p_weight=particle.getWeight();
    }
  }
    
}
