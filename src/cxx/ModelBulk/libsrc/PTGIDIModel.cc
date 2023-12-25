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

  // PoPI::Database pops( "/home/caixx/git/gidiplus/GIDI/Test/pops.xml" );
  // GIDI::Map::Map map( "/home/caixx/git/gidiplus/GIDI/Test/all3T.map", pops );

  GIDI::Construction::Settings construction( GIDI::Construction::ParseMode::MonteCarloContinuousEnergy, 
                                               GIDI::Construction::PhotoMode::nuclearAndAtomic );
 
  GIDI::ProtareSingle *protare = static_cast<GIDI::ProtareSingle *>( map.protare( construction, pops, "n", "He3" ) );


  
  GIDI::Styles::TemperatureInfos temperatures = protare->temperatures( );
  for( GIDI::Styles::TemperatureInfos::const_iterator iter = temperatures.begin( ); iter != temperatures.end( ); ++iter ) {
    std::cout << "label = " << iter->heatedCrossSection( ) << "  temperature = " << iter->temperature( ).value( ) << std::endl;
  }

  std::string label( temperatures[0].heatedCrossSection( ) );


  // MC Part
  MCGIDI::Transporting::MC *MC = new MCGIDI::Transporting::MC( pops, protare->projectile( ).ID( ), &protare->styles( ), label, GIDI::Transporting::DelayedNeutrons::on, 20.0 );
  MC->setNuclearPlusCoulombInterferenceOnly( true );
  MC->sampleNonTransportingParticles( true );

  LUPI::StatusMessageReporting smr1;
  GIDI::Transporting::Particles particles;
  std::set<int> reactionsToExclude;


  MCGIDI::DomainHash domainHash( 4000, 1e-8, 10 );
  MCGIDI::Protare *MCProtare;
  MCProtare = MCGIDI::protareFromGIDIProtare( smr1, *protare, pops, *MC, particles, domainHash, temperatures, reactionsToExclude );



  MCGIDI::Vector<MCGIDI::Protare *> protares( 1 );
  protares[0] = MCProtare;
  MCGIDI::URR_protareInfos URR_protare_infos( protares );


  MCProtare = MCGIDI::protareFromGIDIProtare( smr1, *protare, pops, *MC, particles, domainHash, temperatures, reactionsToExclude );


  for( double energy = 1e-12; energy < 100; energy *= 3 ) {
    int hashIndex = domainHash.index( energy );

    double crossSection = MCProtare->crossSection( URR_protare_infos, hashIndex, 2.586e-08, energy );
    std::cout << "    energy = " << energy << " crossSection = " << crossSection << std::endl;
  }

}

Prompt::GIDIModel::~GIDIModel()
{
  std::cout<<"Destructing GIDIModel " << m_modelName <<std::endl;
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
