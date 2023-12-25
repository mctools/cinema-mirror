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
#include "GIDI.hpp"

#include "LUPI.hpp"
#include "MCGIDI.hpp"

#include <limits>

#include "PTGIDIModel.hh"
// #include "PTRandCanonical.hh"




Prompt::GIDIModel::GIDIModel(const std::string &cfgstring, double bias)
:Prompt::DiscreteModel("GIDI", const_neutron_pgd,
                      std::numeric_limits<double>::min(), 10*Prompt::Unit::eV, bias)
{
  PoPI::Database pops( "/home/caixx/git/fudge/rundir/prompt_data/pops.xml" );
  GIDI::Map::Map map( "/home/caixx/git/fudge/rundir/prompt_data/neutron.map", pops );

  // GIDI::Transporting::Particles particles;
  // std::set<int> reactionsToExclude;
  // LUPI::StatusMessageReporting smr1;


  GIDI::Construction::Settings construction( GIDI::Construction::ParseMode::MonteCarloContinuousEnergy, 
                                               GIDI::Construction::PhotoMode::nuclearAndAtomic );
 
  GIDI::ProtareSingle *protare = static_cast<GIDI::ProtareSingle *>( map.protare( construction, pops, "n", "O16" ) );


  
  GIDI::Styles::TemperatureInfos temperatures = protare->temperatures( );
  for( GIDI::Styles::TemperatureInfos::const_iterator iter = temperatures.begin( ); iter != temperatures.end( ); ++iter ) {
    std::cout << "label = " << iter->heatedCrossSection( ) << "  temperature = " << iter->temperature( ).value( ) << std::endl;
  }



  // m_oriented = m_scat.isOriented();
  // if(Unit::eV != 1.)
  //   PROMPT_THROW(CalcError, "The default unit of NCrystal is eV");
}

Prompt::GIDIModel::~GIDIModel()
{
  std::cout<<"Destructing scattering physics " << m_modelName <<std::endl;
}


double Prompt::GIDIModel::getCrossSection(double ekin) const
{
  return 0;
  // if( m_scat.isOriented() ) {
  //   PROMPT_THROW(CalcError, "direction should be provided for oriented material");
  // }
  // else
  // {
  //   auto xsect = m_scat.crossSectionIsotropic( NCrystal::NeutronEnergy(ekin) );
  //   return xsect.get()*m_bias*Unit::barn;
  // }
}

double Prompt::GIDIModel::getCrossSection(double ekin, const Prompt::Vector &dir) const
{
  return 0;
  // NCrystal::CrossSect xsect;
  // if( m_scat.isOriented() ) {
  //   xsect = m_scat.crossSection( NCrystal::NeutronEnergy(ekin), {dir.x(), dir.y(), dir.z()} );
  // }
  // else
  // {
  //   xsect = m_scat.crossSectionIsotropic( NCrystal::NeutronEnergy(ekin) );
  // }
  // return xsect.get()*m_bias*Unit::barn;
}


void Prompt::GIDIModel::generate(double ekin, const Prompt::Vector &dir, double &final_ekin, Prompt::Vector &final_dir) const
{
  // auto outcome1 = m_scat.sampleScatter( NCrystal::NeutronEnergy(ekin), {dir.x(), dir.y(), dir.z()});
  // final_ekin = outcome1.ekin.get();
  // auto &outdir = outcome1.direction;
  // final_dir.x() = outdir[0];
  // final_dir.y() = outdir[1];
  // final_dir.z() = outdir[2];
}
