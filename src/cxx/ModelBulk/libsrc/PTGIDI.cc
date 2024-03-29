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
#include "MCGIDI_sampling.hpp"
#include <limits>

#include "PTGIDI.hh"

#include <iostream>
#include <iomanip>
#include <functional>
#include "PTMaterialDecomposer.hh"
#include <tuple> // for tuple


inline double getRandNumber(void *obj) 
{
  return Prompt::Singleton<Prompt::SingletonPTRand>::getInstance().generate();
}


Prompt::GIDIModel::GIDIModel(const std::string &name, std::shared_ptr<MCGIDI::Protare> mcprotare,
                             double temperature, 
                             double bias, double frac, double lowerlimt, double upperlimt)
:Prompt::DiscreteModel(name, const_neutron_pgd,
                      lowerlimt,
                      upperlimt, 
                      bias),
m_factory(Prompt::Singleton<Prompt::GIDIFactory>::getInstance()), 
m_mcprotare(mcprotare), 
m_urr_info(nullptr),
m_products(new MCGIDI::Sampling::StdVectorProductHandler()),
m_cacheEkin(0.), 
m_cacheGidiXS(0.),
m_temperature(temperature),
m_frac(frac),
m_input(new MCGIDI::Sampling::Input(false, MCGIDI::Sampling::Upscatter::Model::B) ),
m_elasticReactionIndex(-1)
{ 


  MCGIDI::Vector<MCGIDI::Protare *> protares(1);
  protares[0]= m_mcprotare.get();
  m_urr_info = new MCGIDI::URR_protareInfos(protares);


  m_input->m_temperature = const_boltzmann*temperature/Unit::keV;   // In keV/k;

    // std::cout << "!!!!!!!!! " << temperature_K << " " << const_boltzmann*293.15/Unit::MeV << std::endl;

  int numberOfReactions = m_mcprotare->numberOfReactions();
  std::cout <<"Isotope " << name << " has " << numberOfReactions << " reactions"<< "\n";
     
  for( int i = 0; i < numberOfReactions; ++i ) 
  {
    auto reaction =  m_mcprotare->reaction(i);
    // The type of ENDF_MT can be found at https://t2.lanl.gov/nis/endf/mts.html
    if(reaction->ENDF_MT()==2)
    {
      m_elasticReactionIndex = i;
    }
    std::cout << "Reaction " << i << ", ENDF_MT=" << reaction->ENDF_MT()  << std::endl;
  }

  if(m_elasticReactionIndex < 0)
    PROMPT_THROW2(BadInput, "Elastic scattering reaction (MT2, see https://t2.lanl.gov/nis/endf/mts.html) is missing in " << name);

}

Prompt::GIDIModel::~GIDIModel()
{
  delete m_products;
  delete m_urr_info;
  delete m_input;
  std::cout<<"Destructing GIDIModel " << m_modelName <<std::endl;
}


double Prompt::GIDIModel::getCrossSection(double ekin) const
{
  if (!m_modelvalid.ekinValid(ekin))
    return 0;

  if(ekin==m_cacheEkin)
  {
    return m_cacheGidiXS*m_bias*Unit::barn*m_frac;
  }
  else
  {
    
    m_cacheEkin = ekin;
    double ekin_MeV = ekin*1e-6;
    int hashIndex = m_factory.getHashID(ekin_MeV);
    //temperature unit here is MeV/k, not to confuse by the keV/k unit in m_input
    m_cacheGidiXS = m_mcprotare->crossSection( *m_urr_info, hashIndex, const_boltzmann*m_temperature*1e-6, ekin_MeV ); 
    if( m_factory.NCrystal4Elastic(ekin) ) // This is the energy region for ncrystal to perform scattering calculation
    {
      double mt2xs = m_mcprotare->reactionCrossSection( m_elasticReactionIndex, *m_urr_info, hashIndex, const_boltzmann*m_temperature*1e-6, ekin_MeV );
      m_cacheGidiXS -= mt2xs;
    }
    
    // std::cout << "sampled despition " << m_mcprotare->depositionEnergy( hashIndex, const_boltzmann*m_temperature*1e-6, gidiEnergy )  << "\n";
    return m_cacheGidiXS*m_bias*Unit::barn*m_frac;
  }
}

double Prompt::GIDIModel::getCrossSection(double ekin, const Prompt::Vector &dir) const
{
  return getCrossSection(ekin);
}


void Prompt::GIDIModel::generate(double ekin, const Prompt::Vector &dir, double &final_ekin, Prompt::Vector &final_dir) const
{

  pt_assert_always(ekin==m_cacheEkin);

  double ekin_MeV = ekin*1e-6;


  //fixme, do not care about temperature at the moment

  int hashIndex = m_factory.getHashID(ekin_MeV);

  int reactionIndex(m_elasticReactionIndex);

  if(m_factory.NCrystal4Elastic(ekin))
  {
    // elastic is treated by NCrystal, remove it from here
    while(reactionIndex==m_elasticReactionIndex)
    {
      reactionIndex = m_mcprotare->sampleReaction( *m_urr_info, hashIndex, m_input->m_temperature, ekin_MeV, m_cacheGidiXS, getRandNumber, nullptr );
    }
  }
  else
    reactionIndex = m_mcprotare->sampleReaction( *m_urr_info, hashIndex, m_input->m_temperature, ekin_MeV, m_cacheGidiXS, getRandNumber, nullptr );
  
 
  MCGIDI::Reaction const *reaction = m_mcprotare->reaction( reactionIndex );
  pt_assert_always(m_mcprotare->threshold( reactionIndex ) < ekin_MeV);

  m_products->clear();
  reaction->sampleProducts( m_mcprotare.get(), ekin_MeV, *m_input, getRandNumber, nullptr, *m_products );

  pt_assert_always(m_input->m_reaction==reaction);


  // debug and looking for the MT value for the selected reaction
  // std::cout << "ENDF MT" << reaction->ENDF_MT() <<  ", m_products->size() " <<  m_products->size() << std::endl;
  std::vector<MCGIDI::Sampling::Product> prod_n;


  double totalekin = 0;
  for( std::size_t i = 0; i < m_products->size( ); ++i ) 
  {
    if ((*m_products)[i].m_productIndex==11) //neutron
    {
      prod_n.push_back((*m_products)[i]);
      totalekin += (*m_products)[i].m_kineticEnergy;
    }
      

    // // if(reaction->finalQ(ekin_MeV))
    std::cout << (*m_products)[i].m_productIndex << " " 
    << reaction->finalQ(ekin_MeV) << " " 
    << (*m_products)[i].m_kineticEnergy << "\n";
  }
  //   std::cout <<"incident " << ekin_MeV;
  // std::cout <<", total neutron energy " << totalekin << "\n";

  // All secondary particles that are not simulated by prompt are contributed to the "energy deposition".
  // So, in the case that neutron is the only transporting particle, the energy deposition is calculated as incident neutorn kinetic energy 
  // plus Q and substrcut the total kinetic energy of all the tracking particles. 

  printf("MT%d, deposition %f\n\n", reaction->ENDF_MT(), ekin_MeV+reaction->finalQ(ekin_MeV)-totalekin);

  // std::cout << " total neutrons " << prod_n.size() << "\n";

  // if MC.sampleNonTransportingParticles(true), many of the events are sampled in the centerOfMass
  if(m_input->m_frame == GIDI::Frame::centerOfMass)
  {
    if(m_input->m_sampledType == MCGIDI::Sampling::SampledType::firstTwoBody)
      std::cout << "MCGIDI::Sampling::SampledType::firstTwoBody\n";
    else if(m_input->m_sampledType == MCGIDI::Sampling::SampledType::secondTwoBody)
      std::cout << "MCGIDI::Sampling::SampledType::secondTwoBody\n";
    else if(m_input->m_sampledType == MCGIDI::Sampling::SampledType::uncorrelatedBody)
      std::cout << "MCGIDI::Sampling::SampledType::uncorrelatedBody\n";
    else if(m_input->m_sampledType == MCGIDI::Sampling::SampledType::unspecified)
      std::cout << "MCGIDI::Sampling::SampledType::unspecified\n";
    else if(m_input->m_sampledType == MCGIDI::Sampling::SampledType::photon)
      std::cout << "MCGIDI::Sampling::SampledType::photon\n";
    else
      PROMPT_THROW(CalcError, "unknown m_input->m_sampledType");

  
    // PROMPT_THROW(NotImplemented, "GIDI::Frame::centerOfMass product is not yet implemented");
  }

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
    final_ekin=ENERGYTOKEN_ABSORB;

    for(auto v: prod_n)
    {
        std::cout << "prd idx " << v.m_productIndex << " "
        << v.m_kineticEnergy << std::endl;
    }
    std::cout << std::endl;
    // //Fixme: mcpl writer will be used to record neutron multiplication for keff application
    PROMPT_THROW(NotImplemented, "neutron multiplication is not yet supported");
  }
}

Prompt::GIDIFactory::GIDIFactory()
:m_ctrdata(Singleton<Prompt::CentralData>::getInstance()),
m_pops(new PoPI::Database( m_ctrdata.getGidiPops())),
m_map (new GIDI::Map::Map( m_ctrdata.getGidiMap(), *m_pops )),
m_particles(new GIDI::Transporting::Particles()),
m_construction(new GIDI::Construction::Settings ( GIDI::Construction::ParseMode::all, 
                                              GIDI::Construction::PhotoMode::nuclearAndAtomic )),
m_domainHash(new MCGIDI::DomainHash ( 4000, 1e-8, 20 ) ),
m_reactionsToExclude(std::set<int>()),
m_smr1()
{  
  GIDI::Transporting::Particle neutron(PoPI::IDs::neutron, GIDI::Transporting::Mode::MonteCarloContinuousEnergy);
  m_particles->add(neutron);
  pt_assert_always((*m_pops)["n"] == 11);
 
  if(m_ctrdata.getGammaTransport())
  {
    GIDI::Transporting::Particle photon( PoPI::IDs::photon, GIDI::Transporting::Mode::MonteCarloContinuousEnergy );
    m_particles->add( photon );
  }

  m_pops->print(true);
  // m_pops->isMetaStableAlias();

  // std::cout << "!!!!!!!!!!!!!! "<<(*m_pops)["n"] << std::endl;
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

std::vector<std::shared_ptr<Prompt::GIDIModel>> Prompt::GIDIFactory::createGIDIModel(std::vector<Prompt::IsotopeComposition> vecComp, double bias, double minEKin, double maxEKin) const
{
  std::vector<std::shared_ptr<GIDIModel>> gidimodels;
  MCGIDI::Vector<MCGIDI::Protare *> protares(vecComp.size());
  std::vector<std::tuple<std::shared_ptr<MCGIDI::ProtareSingle>, std::string, double, double>> singleProtares;

  size_t i = 0;

  // fixme: make shared pointer map to cache MCGIDI::ProtareSingle for repeated isotopes
  // the key should be the label (i.e. iter->heatedCrossSection( )) plus the  isotope name
  for(const auto& isotope : vecComp)
  {
    const std::string &name = isotope.name;
    double frac = isotope.frac;

    if(!m_map->isProtareAvailable( PoPI::IDs::neutron, name))
    {
      PROMPT_THROW2(DataLoadError, "GIDIFactory failed to load data for " << name);
    }
    auto *gidiprotare =  (GIDI::Protare *) m_map->protare( *m_construction, *m_pops, "n", name, "", "", true, true ) ;
    
    GIDI::Styles::TemperatureInfos temperatures = gidiprotare->temperatures( );
    for( GIDI::Styles::TemperatureInfos::const_iterator iter = temperatures.begin( ); iter != temperatures.end( ); ++iter ) {
      std::cout << "label = " << iter->heatedCrossSection( ) << "  temperature = " << iter->temperature( ).value( ) << std::endl;
    }

    std::string label( temperatures[0].heatedCrossSection( ) );
    double temperature_K = temperatures[0].temperature( ).value() * Unit::MeV / const_boltzmann; 

    MCGIDI::Transporting::MC MC ( *m_pops, gidiprotare->projectile( ).ID( ), &gidiprotare->styles( ), label, GIDI::Transporting::DelayedNeutrons::on, 20.0 );
    // MC.setNuclearPlusCoulombInterferenceOnly( false );
    MC.sampleNonTransportingParticles( m_ctrdata.getGidiSampleNTP() );
    // MC.set_ignoreENDF_MT5(true);

   

    if( gidiprotare->protareType( ) != GIDI::ProtareType::single ) {
        PROMPT_THROW(CalcError, "ProtareType must be single");
    }

    auto mcProtare = std::make_shared<MCGIDI::ProtareSingle>(*m_smr1, static_cast<GIDI::ProtareSingle const &>( *gidiprotare), *m_pops, MC, 
                                                             *m_particles, *m_domainHash, temperatures, m_reactionsToExclude );

    // singleProtares.push_back(std::make_tuple(mcProtare, name, temperature_K, frac));
    gidimodels.emplace_back(std::make_shared<GIDIModel>(name, mcProtare, temperature_K, bias, frac, minEKin, maxEKin));

    i++;
    delete gidiprotare;
  }

  // auto URR_protare_infos = std::make_shared<MCGIDI::URR_protareInfos>();
  // for(auto s : singleProtares)
  //   gidimodels.emplace_back(std::make_shared<GIDIModel>(std::get<1>(s), std::get<0>(s), std::get<2>(s), bias, std::get<3>(s), minEKin, maxEKin));

  return (gidimodels);
}
