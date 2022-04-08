#ifndef Prompt_ScororNeutronSq_hh
#define Prompt_ScororNeutronSq_hh

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

#include "PromptCore.hh"
#include "PTScoror.hh"

namespace Prompt {

  class ScororNeutronSq  : public Scoror1D {
  public:
    ScororNeutronSq(const std::string &name, const Vector &samplePos, const Vector &refDir,
      double sourceSampleDist, double qmin, double qmax, unsigned numbin,
      ScororType styp=Scoror::ENTRY, bool linear=true);
    virtual ~ScororNeutronSq();
    virtual void scoreLocal(const Vector &vec, double w) override;
    virtual void score(Particle &particle) override;
    virtual void score(Particle &particle, const DeltaParticle &dltpar) override;
  private:
    const Vector m_samplePos, m_refDir;
    const double m_sourceSampleDist;
    bool m_kill;
  };
}
#endif
