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

#include "PTMaterialPhysics.hh"
#include <cmath>
#include <limits>
#include "NCrystal/NCrystal.hh"

Prompt::MaterialPhysics::MaterialPhysics()
:m_rng(Singleton<SingletonPTRand>::getInstance()),
m_compModel(std::make_unique<CompoundModel>()),
m_numdensity(0.) { }

Prompt::MaterialPhysics::~MaterialPhysics() { }

double Prompt::MaterialPhysics::macroCrossSection(const Prompt::Particle &particle) const
{
  return m_numdensity*m_compModel->totalCrossSection(particle.getEKin(), particle.getDirection());
}

void Prompt::MaterialPhysics::sampleFinalState(Prompt::Particle &particle, double stepLength, bool hitWall) const
{
  double final_ekin;
  Vector final_dir;
  m_compModel->generate(particle.getEKin(), particle.getDirection(), final_ekin, final_dir);
  particle.setEKin(final_ekin);
  particle.setDirection(final_dir);
  if(!stepLength)
    particle.scaleWeight(m_compModel->calculateWeight(stepLength*m_numdensity, hitWall));
}

double Prompt::MaterialPhysics::sampleStepLength(const Prompt::Particle &particle) const
{
  double mxs = macroCrossSection(particle);
  if(mxs)
  {
    return -log(m_rng.generate())/mxs;
  }
  else
  {
    return std::numeric_limits<double>::max();
  }
}


double Prompt::MaterialPhysics::calNumDensity(const std::string &cfg)
{
  NCrystal::MatCfg matcfg(cfg);
  auto info = NCrystal::createInfo(matcfg);
  if(info->hasNumberDensity())
    return info->getNumberDensity().get() / Unit::Aa3;
  else
  {
    PROMPT_THROW2(CalcError, "material has no number density " << cfg);
    return 0.;
  }
}

void Prompt::MaterialPhysics::addComposition(const std::string &cfg, double bias)
{
  m_compModel->addPhysicsModel(cfg, bias);
  m_numdensity += calNumDensity(cfg);
}
