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

#include "PTGIDI.hh"

#include <iostream>
#include <iomanip>
#include <functional>


inline double getRandNumber(void *obj) 
{
  return Prompt::Singleton<Prompt::SingletonPTRand>::getInstance().generate();
}


Prompt::GIDIModel::GIDIModel(const std::string &name, std::shared_ptr<MCGIDI::Protare> mcprotare,
                             std::shared_ptr<MCGIDI::URR_protareInfos> urr_info, double bias)
:Prompt::DiscreteModel("GIDI", const_neutron_pgd,
                      std::numeric_limits<double>::min(), 
                      10*Prompt::Unit::eV, bias),
m_factory(Prompt::Singleton<Prompt::GIDIFactory>::getInstance()), 
m_mcprotare(mcprotare), 
m_urr_info(urr_info),
m_products(new MCGIDI::Sampling::StdVectorProductHandler()),
m_cacheEkin(0.), 
m_cacheGidiXS(0.)
{ 
  int numberOfReactions = m_mcprotare->numberOfReactions();
  std::cout <<"Isotope " << name << " has " << numberOfReactions << " reactions"<< "\n";
     
  for( int i = 0; i < numberOfReactions; ++i ) 
  {
    auto reaction =  m_mcprotare->reaction(i);
    std::cout << "Reaction " << i << ", ENDF_MT=" << reaction->ENDF_MT()  << std::endl;
  }
  // The type of ENDF_MT can be found at https://t2.lanl.gov/nis/endf/mts.html
}

Prompt::GIDIModel::~GIDIModel()
{
  delete m_products;
  std::cout<<"Destructing GIDIModel " << m_modelName <<std::endl;
}


double Prompt::GIDIModel::getCrossSection(double ekin) const
{
  if(ekin==m_cacheEkin)
  {
    return m_cacheGidiXS*m_bias*Unit::barn;
  }
  else
  {
    m_cacheEkin = ekin;
    double gidiEnergy = ekin*1e-6;
    int hashIndex = m_factory.getHashID(gidiEnergy);
    m_cacheGidiXS = m_mcprotare->crossSection( *m_urr_info.get(), hashIndex, 2.586e-08, gidiEnergy );
    return m_cacheGidiXS*m_bias*Unit::barn;
  }
}

double Prompt::GIDIModel::getCrossSection(double ekin, const Prompt::Vector &dir) const
{
  return getCrossSection(ekin);
}


void Prompt::GIDIModel::generate(double ekin, const Prompt::Vector &dir, double &final_ekin, Prompt::Vector &final_dir) const
{
  pt_assert_always(ekin==m_cacheEkin);

  double energy = ekin*1e-6;

  MCGIDI::Sampling::Input input( true, MCGIDI::Sampling::Upscatter::Model::none );
  input.m_temperature = 2.58e-5;                                                     // In keV/k;
  //fixme, do not care about temperature at the moment

  int hashIndex = m_factory.getHashID(energy);

  int reactionIndex = m_mcprotare->sampleReaction( *m_urr_info.get(), hashIndex, input.m_temperature, energy, m_cacheGidiXS, getRandNumber, nullptr );
  MCGIDI::Reaction const *reaction = m_mcprotare->reaction( reactionIndex );
  
  // ///////////////////////////////////////////////////////////////////////////////////
  // // checking the cross section contributions from each reaction
  // char Str[128];

  // int numberOfReactions = m_mcprotare->numberOfReactions();
  // for( int i1 = 0; i1 < numberOfReactions; ++i1 ) {
  //     double reactionCrossSection = m_mcprotare->reactionCrossSection( i1, *m_urr_info.get(), hashIndex, input.m_temperature, energy );
  //     sprintf( Str, " %9.6f", reactionCrossSection / crossSection );
  //     std::cout << Str;
  // }
  // std::cout << std::endl;
  // ///////////////////////////////////////////////////////////////////////////////////

  pt_assert_always(m_mcprotare->threshold( reactionIndex ) < energy);
  m_products->clear();
  reaction->sampleProducts( m_mcprotare.get(), energy, input, getRandNumber, nullptr, *m_products );


  std::cout << "ENDF MT" << reaction->ENDF_MT() <<  ", m_products->size() " <<  m_products->size() << std::endl;
  std::vector<MCGIDI::Sampling::Product> prod_n;
  for( std::size_t i = 0; i < m_products->size( ); ++i ) 
  {
    std::cout << (*m_products)[i].m_productIndex << "\n";
    prod_n.push_back((*m_products)[i]);
  }

  std::cout << " total neutrons " << prod_n.size() << "\n";
  // Neutron die as absorption
  if(prod_n.size()==0)
  {
    final_ekin=ENERGYTOKEN_ABSORB;
  }
  else if(prod_n.size()==1)
  {
    final_ekin = prod_n[0].m_kineticEnergy*1e6;
    final_dir.x() = prod_n[0].m_px_vx;
    final_dir.y() = prod_n[0].m_py_vy;
    final_dir.z() = prod_n[0].m_pz_vz;
    final_dir.setMag(1.);
  }
  else
  {
    //Fixme: mcpl writer will be used to record neutron multiplication for keff application
    PROMPT_THROW(NotImplemented, "neutron multiplication is not yet supported");
  }

  
  // auto outcome1 = m_scat.sampleScatter( NCrystal::NeutronEnergy(ekin), {dir.x(), dir.y(), dir.z()});
  // final_ekin = outcome1.ekin.get();
  // auto &outdir = outcome1.direction;
  // final_dir.x() = outdir[0];
  // final_dir.y() = outdir[1];
  // final_dir.z() = outdir[2];
}


Prompt::GIDIFactory::GIDIFactory()
:m_pops(new PoPI::Database( "/home/caixx/git/fudge/rundir/prompt_data/pops.xml" )),
m_map (new GIDI::Map::Map( "/home/caixx/git/fudge/rundir/prompt_data/neutron.map", *m_pops )),
m_particles(new GIDI::Transporting::Particles()),
m_construction(new GIDI::Construction::Settings ( GIDI::Construction::ParseMode::MonteCarloContinuousEnergy, 
                                              GIDI::Construction::PhotoMode::nuclearAndAtomic )),
m_domainHash(new MCGIDI::DomainHash ( 4000, 1e-8, 10 ) ),
m_reactionsToExclude(std::set<int>())
{  
  GIDI::Transporting::Particle neutron(PoPI::IDs::neutron, GIDI::Transporting::Mode::MonteCarloContinuousEnergy);
  m_particles->add(neutron);
  // GIDI::Transporting::Particle photon( PoPI::IDs::photon, GIDI::Transporting::Mode::MonteCarloContinuousEnergy);
  // particles.add(photon);

  // m_pops->print(true);

  // std::cout << (*m_pops)["n"] << std::endl;
  // std::cout << (*m_pops)["He3"] << std::endl;
  // std::cout << (*m_pops)["O16"] << std::endl;
  // std::cout << (*m_pops)["O16"] << std::endl;


}

Prompt::GIDIFactory::~GIDIFactory()
{
  delete m_pops;
  delete m_map;
  delete m_particles;
  delete m_construction;
  delete m_domainHash;
}

int Prompt::GIDIFactory::getHashID(double energy) const
{
  return m_domainHash->index(energy);
}

bool Prompt::GIDIFactory::available() const
{
  return true;
}


std::shared_ptr<Prompt::GIDIModel> Prompt::GIDIFactory::createGIDIModel(const std::string &name, double bias) const
{
  if(!m_map->isProtareAvailable( PoPI::IDs::neutron, name))
  {
    PROMPT_THROW2(DataLoadError, "GIDIFactory failed to load data for " << name);
  }

  auto *protare = m_map->protare( *m_construction, *m_pops, "n", name ) ;
  
  GIDI::Styles::TemperatureInfos temperatures = protare->temperatures( );
  for( GIDI::Styles::TemperatureInfos::const_iterator iter = temperatures.begin( ); iter != temperatures.end( ); ++iter ) {
    std::cout << "label = " << iter->heatedCrossSection( ) << "  temperature = " << iter->temperature( ).value( ) << std::endl;
  }

  std::string label( temperatures[0].heatedCrossSection( ) );

  // MC Part
  MCGIDI::Transporting::MC *MC = new MCGIDI::Transporting::MC( *m_pops, protare->projectile( ).ID( ), &protare->styles( ), label, GIDI::Transporting::DelayedNeutrons::on, 20.0 );
  MC->setNuclearPlusCoulombInterferenceOnly( false );
  MC->sampleNonTransportingParticles( false );
  

  LUPI::StatusMessageReporting smr1;
  if( protare->protareType( ) != GIDI::ProtareType::single ) {
      PROMPT_THROW(CalcError, "ProtareType must be single");
  }
  auto MCProtare = std::make_shared<MCGIDI::ProtareSingle>(smr1, static_cast<GIDI::ProtareSingle const &>( *protare), *m_pops, *MC, *m_particles, *m_domainHash, temperatures, m_reactionsToExclude );

  MCGIDI::Vector<MCGIDI::Protare *> protares( 1 );
  protares[0] = MCProtare.get();
  auto URR_protare_infos = std::make_shared<MCGIDI::URR_protareInfos>(protares);

  return std::make_shared<GIDIModel>(name, MCProtare, URR_protare_infos, bias);
}