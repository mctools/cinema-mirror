#ifndef Prompt_ScorerMultiScatMixin_hh
#define Prompt_ScorerMultiScatMixin_hh

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
#include "PTScorerMultiScat.hh"

namespace Prompt {
  template <typename T>
  class ScorerMultiScatMixin {
  public:
    // Constructor to initialize the log level
    ScorerMultiScatMixin(const ScorerMultiScat* scatterCounter, int scatNumReq=-1) : 
      m_scatterNumberRequired(scatNumReq), m_scatterCounter(scatterCounter) {}

    void addMultiScatter(const Prompt::ScorerMultiScat* scatterCounter, int scatNumReq=-1 ) 
    {
      m_scatterCounter=scatterCounter;
      m_scatterNumberRequired = scatNumReq;
      std::cout << "Scattering number req: " << m_scatterNumberRequired << std::endl;
    }

    bool rightScatterNumber() const {
      return (m_scatterNumberRequired && m_scatterCounter!=nullptr) ? 
      m_scatterCounter->getScatNumber()==m_scatterNumberRequired : true;
    }

  protected:
    const ScorerMultiScat* m_scatterCounter;
    int m_scatterNumberRequired;
  };

}
#endif
