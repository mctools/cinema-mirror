#ifndef Prompt_Scorer_hh
#define Prompt_Scorer_hh

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
#include "PTParticle.hh"
#include "PTHist1D.hh"
#include "PTHist2D.hh"
// #include "PTActiveVolume.hh"
// #include "PTSingleton.hh"

namespace Prompt {
   class ActiveVolume;
  class Scorer {
  public:
    enum class ScorerType {SURFACE, ENTRY, PROPAGATE, EXIT, ENTRY2EXIT, ABSORB};
  public:
    Scorer(const std::string& name, ScorerType type, int groupid=0) ;
    virtual ~Scorer() {std::cout<<"Destructing Scorer " << m_name <<std::endl;};
    const std::string &getName() const { return m_name; }
    ScorerType getType() const { return m_type; }
    virtual void score(Particle &particle) = 0;
    virtual void save_mcpl() = 0;
    virtual const HistBase* getHist() const = 0 ; 
  protected:
    const std::string m_name;
    const ScorerType m_type;
    const int m_groupid;
    ActiveVolume &m_activeVolume; 
    
  };

  class Scorer1D : public Scorer {
  public:
    Scorer1D(const std::string& name, ScorerType type, std::unique_ptr<Hist1D> hist, int groupid=0)
    : Scorer(name, type, groupid), m_hist(std::move(hist)) {};
    virtual ~Scorer1D() {  }
    void save_mcpl() override { m_hist->save(m_name); }
    const HistBase* getHist() const override  { return dynamic_cast<const HistBase*>(m_hist.get()); }
  protected:
    std::unique_ptr<Hist1D> m_hist;
  };

  class Scorer2D : public Scorer {
  public:
    Scorer2D(const std::string& name, ScorerType type, std::unique_ptr<Hist2D> hist, int groupid=0)
    : Scorer(name, type, groupid), m_hist(std::move(hist)) {};
    virtual ~Scorer2D() {  }
    void save_mcpl() override { m_hist->save(m_name); }
    const HistBase* getHist() const override  { return dynamic_cast<const HistBase*>(m_hist.get()); }

  protected:
    std::unique_ptr<Hist2D> m_hist;
  };
}

#endif
