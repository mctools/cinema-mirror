#ifndef Prompt_Track_hh
#define Prompt_Track_hh

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

#include <vector>
#include "PTVector.hh"
#include "PTParticle.hh"

namespace Prompt {
  class Particle;

  struct SpaceTime {
    Vector pos;
    double time; // negative time means deleted particle
    //region ???
  };

  class Track : public Particle {
  public:
    Track(Particle &&particle, size_t eventid, size_t motherid, bool saveSpaceTime=true);
    virtual  ~Track();
    virtual void moveForward(double length) override;

  private:
    void update(double length=0);
    size_t m_eventid, m_motherid; //the first particle's m_motherid is zero
    double m_totLength;
    bool m_saveSpaceTime;
    std::vector<SpaceTime> m_spacetime;
  };
}

inline void Prompt::Track::moveForward(double length)
{
  Particle::moveForward(length);
  update(length);
}

inline void Prompt::Track::update(double length)
{
  m_totLength += length;
  if(m_saveSpaceTime)
    m_spacetime.emplace_back(SpaceTime{m_pos, m_time} );
}

#endif
