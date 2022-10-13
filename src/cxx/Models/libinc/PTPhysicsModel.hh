#ifndef Prompt_PhysicsModel_hh
#define Prompt_PhysicsModel_hh

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

#include <string>
#include <memory>
#include "PromptCore.hh"
#include "PTVector.hh"

#include "NCrystal/NCrystal.hh"
#include "PTRandCanonical.hh"

namespace Prompt {

  class PhysicsModel {
  public:
    PhysicsModel(const std::string &name);
    PhysicsModel(const std::string &name, unsigned gdp, double emin, double emax);
    virtual ~PhysicsModel() {};

    const std::string &getName() { return m_modelName; }
    bool isOriented();
    void getEnergyRange(double &ekinMin, double &ekinMax) ;
    void setEnergyRange(double ekinMin, double ekinMax);

    virtual bool applicable(unsigned pgd) const;
    virtual bool applicable(unsigned pgd, double ekin) const;
    virtual double getCrossSection(double ekin) const;
    virtual double getCrossSection(double ekin, const Vector &dir) const;

    // final_ekin -1., propose kill because of an absorb event
    // final_ekin -2., propose kill because of a biasing event
    virtual void generate(double ekin, const Vector &dir, double &final_ekin, Vector &final_dir, double &scaleWeight) const = 0;

  protected:
    std::string m_modelName;
    unsigned m_supportPGD;
    double m_minEkin, m_maxEkin;
    bool m_oriented;
    SingletonPTRand &m_rng;
  };

}

#endif
