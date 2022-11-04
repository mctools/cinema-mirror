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

#include "PTPhysicsModel.hh"

Prompt::PhysicsBase::PhysicsBase(const std::string &name)
 :m_modelName(name), m_oriented(false), m_rng(Singleton<SingletonPTRand>::getInstance()) {};

Prompt::PhysicsBase::PhysicsBase(const std::string &name, unsigned gdp,
             double emin, double emax)
 :m_modelName(name), m_supportPGD(gdp), m_minEkin(emin),
  m_maxEkin(emax), m_oriented(false), m_rng(Singleton<SingletonPTRand>::getInstance())  {};

Prompt::PhysicsModel::PhysicsModel(const std::string &name)
:PhysicsBase(name) {}

Prompt::PhysicsModel::PhysicsModel(const std::string &name, unsigned gdp,
             double emin, double emax)
:PhysicsBase(name, gdp, emin, emax) { }

bool Prompt::PhysicsBase::applicable(unsigned pgd) const
{ return m_supportPGD==pgd; }

bool Prompt::PhysicsBase::isOriented()
{return m_oriented;}

void Prompt::PhysicsBase::getEnergyRange(double &ekinMin, double &ekinMax)
{
  m_minEkin = ekinMin;
  m_maxEkin = ekinMax;
};

void Prompt::PhysicsBase::setEnergyRange(double ekinMin, double ekinMax)
{
  ekinMin = m_minEkin;
  ekinMax = m_maxEkin;
};

bool Prompt::PhysicsBase::applicable(unsigned pgd, double ekin) const
{
  return pgd==m_supportPGD && (ekin > m_minEkin && ekin < m_maxEkin);
}

double Prompt::PhysicsBase::getCrossSection(double ekin) const
{
  PROMPT_THROW(LogicError, "getCrossSection is not impletmented ")
  return 0.;
}

double Prompt::PhysicsBase::getCrossSection(double ekin, const Vector &dir) const
{
  PROMPT_THROW(LogicError, "getCrossSection is not impletmented ")
  return 0.;
}
