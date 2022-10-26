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

#include <limits>

#include "PTRotatingObj.hh"
#include "PTUnitSystem.hh"
#include "PTRandCanonical.hh"


Prompt::RotatingObj::RotatingObj(const std::string &cfgstringAsName, const Vector &dir, const Vector &point, double rotFreq)
:Prompt::DiscreteModel(cfgstringAsName, const_neutron_pgd,
                      std::numeric_limits<double>::min(), 10*Prompt::Unit::eV, 1.),
                      m_dir(dir), m_point(point), m_angularfreq(2*M_PI*rotFreq)

{
  //fixme use m_dir.normalise() to make sure the accuracy of the conversion
  if(!floateq(m_dir.mag(),1., 1e-5, 1e-5))
    PROMPT_THROW(BadInput, "direction must be a unit vector");
}


Prompt::RotatingObj::~RotatingObj()
{
  std::cout<<"Destructing RotatingObj " << m_modelName <<std::endl;
}


double Prompt::RotatingObj::getCrossSection(double ekin) const
{
  return 0;
}

double Prompt::RotatingObj::getCrossSection(double ekin, const Prompt::Vector &dir) const
{
  return 0;
}


void Prompt::RotatingObj::generate(double ekin, const Prompt::Vector &dir, double &final_ekin, Prompt::Vector &final_dir) const
{
}
