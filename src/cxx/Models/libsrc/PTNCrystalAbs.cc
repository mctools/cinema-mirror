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

#include "PTNCrystalAbs.hh"
#include "PTUnitSystem.hh"
#include "PTRandCanonical.hh"


Prompt::NCrystalAbs::NCrystalAbs(const std::string &cfgstring, double bias)
:Prompt::DiscreteModel(cfgstring, const_neutron_pgd,
                      std::numeric_limits<double>::min(), 10*Prompt::Unit::eV, bias),
                      m_abs(NCrystal::createAbsorption(cfgstring))
{
  if( m_abs.isOriented() ) {
    PROMPT_THROW(CalcError, "Absorption process is not oriented");
  }
}

Prompt::NCrystalAbs::~NCrystalAbs()
{
  std::cout<<"Destructing absorption physics " << m_modelName <<std::endl;
}


double Prompt::NCrystalAbs::getCrossSection(double ekin) const
{
    return m_abs.crossSectionIsotropic(NCrystal::NeutronEnergy(ekin)).get()*Unit::barn*m_bias;
}

double Prompt::NCrystalAbs::getCrossSection(double ekin, const Prompt::Vector &dir) const
{
  NCrystal::CrossSect xsect = m_abs.crossSectionIsotropic( NCrystal::NeutronEnergy(ekin) );
  return xsect.get()*Unit::barn*m_bias;
}


void Prompt::NCrystalAbs::generate(double ekin, const Prompt::Vector &dir, double &final_ekin, Prompt::Vector &final_dir) const
{
  final_ekin=-1.;
}
