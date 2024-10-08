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

#include "PTPhysicsModel.hh"

Prompt::PhysicsBase::PhysicsBase(const std::string &name, int gdp,
             double emin, double emax)
 :m_modelName(name), m_modelvalid{gdp, emin, emax},
  m_oriented(false), m_rng(Singleton<SingletonPTRand>::getInstance()) {};

Prompt::PhysicsModel::PhysicsModel(const std::string &name, unsigned gdp,
             double emin, double emax)
:PhysicsBase(name, gdp, emin, emax), m_res()  { }


bool Prompt::PhysicsBase::isOriented()
{return m_oriented;}

double Prompt::PhysicsBase::getCrossSection(double ekin) const
{
  PROMPT_THROW(NotImplemented, "PhysicsBase::getCrossSection is not impletmented ")
  return 0.;
}

double Prompt::PhysicsBase::getCrossSection(double ekin, const Vector &dir) const
{
  PROMPT_THROW(NotImplemented, "PhysicsBase::getCrossSection is not impletmented ")
  return 0.;
}
