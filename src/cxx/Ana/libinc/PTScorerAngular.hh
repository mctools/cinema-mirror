#ifndef Prompt_ScorerAngular_hh
#define Prompt_ScorerAngular_hh

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

#include "PromptCore.hh"
#include "PTScorer1D.hh"

namespace Prompt {

  class ScorerAngular  : public Scorer1D {
  public:
    ScorerAngular(const std::string &name, 
      double mu_min, double mu_max, unsigned numbin, unsigned int pdg,
      Vector refDir={0.,0.,1.},
      ScorerType stype=Scorer::ScorerType::ENTRY, bool inDegree=false, bool linear=true);
    virtual ~ScorerAngular();
    virtual void score(Particle &particle) override;
  protected:
    const Vector m_refDir;
    bool m_inDegree;
  };
}
#endif
