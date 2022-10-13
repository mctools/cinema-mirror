#ifndef Prompt_CompoundModel_hh
#define Prompt_CompoundModel_hh

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
#include "PTRandCanonical.hh"
#include "PTDiscreteModel.hh"

namespace Prompt {

  class PhysicsModel;

  struct XSCache {
    double ekin;
    Vector dir;
    std::vector<double> cache_xs;
    std::vector<double> bias;
    double selectedBias;
    double tot;
  };

  // This class is used to represent a collection of models for a material.
  // Only discrete models for now.

  class CompoundModel  {
  public:
    CompoundModel();
    virtual ~CompoundModel();

    double totalCrossSection(double ekin, const Vector &dir) const;
    void sample(double ekin, const Vector &dir, double &final_ekin, Vector &final_dir, double &scaleWeight) const;

    void addPhysicsModel(const std::string &cfg, double bias=1.);
    bool sameInquiryAsLastTime(double ekin, const Vector &dir) const;

    double calculateWeight(double lengthRho, bool selBiase);

  private:
    std::vector<std::shared_ptr<DiscreteModel> > m_models;
    mutable XSCache m_cache;
    bool m_oriented;

    SingletonPTRand &m_rng;

    //fixme: cache
  };
}

inline bool Prompt::CompoundModel::sameInquiryAsLastTime(double ekin, const Vector &dir) const
{
  return m_oriented ? (m_cache.ekin==ekin && m_cache.dir == dir) : m_cache.ekin==ekin;
}

#endif
