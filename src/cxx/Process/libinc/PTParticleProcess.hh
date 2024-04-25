#ifndef Prompt_ParticleProcess_hh
#define Prompt_ParticleProcess_hh

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

#include <string>
#include "PromptCore.hh"
#include "PTModelCollection.hh"
#include "PTParticle.hh"

namespace Prompt {
  class ParticleProcess  {
  public:
    ParticleProcess(const std::string& name, int pdg = const_neutron_pgd);
    const std::string& getName() const { return m_name; }
    virtual ~ParticleProcess();
    ModelCollection* getModelCollection() {return m_compModel.get(); }
    double getNumDensity() {return m_numdensity; }

    double sampleStepLength(const Prompt::Particle &particle) const;
    void sampleFinalState(Prompt::Particle &particle, double stepLength=0., bool hitWall=false) const;
    void cfgPhysicsModel(const std::string &cfg);
    bool containOrentied() const { return m_compModel->containOriented(); }

  private:
    double macroCrossSection(const Prompt::Particle &particle) const;
    std::string m_name;
    SingletonPTRand &m_rng;
    std::unique_ptr<ModelCollection> m_compModel;
    double m_numdensity;

  };

}

#endif
