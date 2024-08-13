#ifndef Prompt_Scorer2D_hh
#define Prompt_Scorer2D_hh

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

#include "PTScorer.hh"
#include "PTScorerMultiScat.hh"
#include "PTMultiScatMixin.hh"

namespace Prompt {
  
  class Scorer2D : public Scorer, public MultiScatMixin<Scorer2D> {
  public:
    Scorer2D(const std::string& name, ScorerType type, std::unique_ptr<Hist2D> hist, unsigned int pdg=0, int groupid=0)
    : Scorer(name, type, groupid), MultiScatMixin(nullptr, -1), m_hist(std::move(hist)) {};
    virtual ~Scorer2D() {  }
    void save_mcpl() override { m_hist->save(m_name); }
    const HistBase* getHist() const override  { return dynamic_cast<const HistBase*>(m_hist.get()); }
    virtual bool rightScorer(const Particle &particle) const override { return Scorer::rightScorer(particle) && rightScatterNumber(); };

  protected:
    std::unique_ptr<Hist2D> m_hist;
  };

}


#endif
