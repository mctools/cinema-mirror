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

#include <limits>

#include "PTNCrystalScat.hh"
#include "PTUnitSystem.hh"
#include "PTRandCanonical.hh"


Prompt::NCrystalScat::NCrystalScat(const std::string &cfgstring, double bias)
:Prompt::DiscreteModel(cfgstring, const_neutron_pgd,
                      std::numeric_limits<double>::min(), 10*Prompt::Unit::eV, bias),
                      m_scat(NCrystal::createScatter(cfgstring))
{
  m_oriented = m_scat.isOriented();
  if(Unit::eV != 1.)
    PROMPT_THROW(CalcError, "The default unit of NCrystal is eV");
}

Prompt::NCrystalScat::~NCrystalScat()
{
  std::cout<<"Destructing scattering physics " << m_modelName <<std::endl;
}


double Prompt::NCrystalScat::getCrossSection(double ekin) const
{
  if( m_scat.isOriented() ) {
    PROMPT_THROW(CalcError, "direction should be provided for oriented material");
  }
  else
  {
    auto xsect = m_scat.crossSectionIsotropic( NCrystal::NeutronEnergy(ekin) );
    return xsect.get()*m_bias*Unit::barn;
  }
}

double Prompt::NCrystalScat::getCrossSection(double ekin, const Prompt::Vector &dir) const
{
  NCrystal::CrossSect xsect;
  if( m_scat.isOriented() ) {
    xsect = m_scat.crossSection( NCrystal::NeutronEnergy(ekin), {dir.x(), dir.y(), dir.z()} );
  }
  else
  {
    xsect = m_scat.crossSectionIsotropic( NCrystal::NeutronEnergy(ekin) );
  }
  return xsect.get()*m_bias*Unit::barn;
}


void Prompt::NCrystalScat::generate(double ekin, const Prompt::Vector &dir, double &final_ekin, Prompt::Vector &final_dir) const
{
  auto outcome1 = m_scat.sampleScatter( NCrystal::NeutronEnergy(ekin), {dir.x(), dir.y(), dir.z()});
  final_ekin = outcome1.ekin.get();
  auto &outdir = outcome1.direction;
  final_dir.x() = outdir[0];
  final_dir.y() = outdir[1];
  final_dir.z() = outdir[2];
}
