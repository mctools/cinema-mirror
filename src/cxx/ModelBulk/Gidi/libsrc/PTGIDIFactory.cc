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

#include <iostream>
#include <iomanip>
#include <functional>
#include "PTMaterialDecomposer.hh"
#include "PTGIDIModel.hh"

#include "PTGIDIFactory.hh"

Prompt::GIDIFactory::GIDIFactory()
:m_ctrdata(Singleton<Prompt::GidiSetting>::getInstance()),
m_pops(new PoPI::Database( m_ctrdata.getGidiPops())),
m_map (new GIDI::Map::Map( m_ctrdata.getGidiMap(), *m_pops )),
m_particles(new GIDI::Transporting::Particles()),
m_construction(new GIDI::Construction::Settings ( GIDI::Construction::ParseMode::all, 
                                              GIDI::Construction::PhotoMode::nuclearAndAtomic )),
m_domainHash(new MCGIDI::DomainHash ( 4000, 1e-8, 200 ) ),
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

  // m_pops->print(true);
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

// todo: fixme: factory should pass the shared pointer of ncrystal sca to the gidi model, alone with the elastic the hold. below the threslhold replace the xs of mt2, but cache the  total xs calculate by mt2 for sampling reaction. if mt2 is leaked at the end, use the mcstas for secondary.
// after that manager should cache all shared pointed gidi models. the isotope should be re used by a key name, the ncsca should be updated.

std::vector<std::shared_ptr<Prompt::GIDIModel>> Prompt::GIDIFactory::createNeutronGIDIModel(const std::vector<Prompt::IsotopeComposition> &vecComp, 
double bias, double elasticThreshold, double minEKin, double maxEKin) const
{
  std::vector<std::shared_ptr<GIDIModel>> gidimodels;
  MCGIDI::Vector<MCGIDI::Protare *> protares(vecComp.size());
  std::vector<std::tuple<std::shared_ptr<MCGIDI::ProtareSingle>, std::string, double, double>> singleProtares;
  
  // auto gdcomp = new GIDI::ProtareComposite( *m_construction);
  // auto mcpro_comp = new MCGIDI::ProtareComposite();

  // fixme: make shared pointer map to cache MCGIDI::ProtareSingle for repeated isotopes
  // the key should be the label (i.e. iter->heatedCrossSection( )) plus the  isotope name
  for(const auto& isotope : vecComp)
  {
    //fixme:: factory that can cache the model from isotope info
    const std::string &name = isotope.name;
    double frac = isotope.frac;

    if(!m_map->isProtareAvailable( PoPI::IDs::neutron, name))
    {
      PROMPT_THROW2(DataLoadError, "GIDIFactory createNeutronGIDIModel failed to load data for " << name << ". ");
    }
    auto *gidiprotare =  (GIDI::Protare *) m_map->protare( *m_construction, *m_pops, "n", name, "", "", true, true ) ;
    // gdcomp->append(gidiprotare);

    std::cout << "Using data file " << gidiprotare->realFileName( ) << std::endl;

    auto delay = GIDI::Transporting::DelayedNeutrons::on;
    if( !gidiprotare->isDelayedFissionNeutronComplete( ) ) 
    {
        std::cout << "WARNING: delayed neutron fission data for "<< name<< " are incomplete and are not included." << std::endl;
        delay = GIDI::Transporting::DelayedNeutrons::off;
    }
    else
      std::cout << "Delayed neutron fission data for "<< name<< " are included." << std::endl;

    GIDI::Transporting::Settings incompleteParticlesSetting( gidiprotare->projectile( ).ID( ),  delay);
    std::set<std::string> incompleteParticles;
    gidiprotare->incompleteParticles( incompleteParticlesSetting, incompleteParticles );
    std::cout << "# List of incomplete particles:";
    for( auto iter = incompleteParticles.begin( ); iter != incompleteParticles.end( ); ++iter ) {
        std::cout << " " << *iter;
    }
    std::cout << std::endl;


    
    GIDI::Styles::TemperatureInfos temperatures = gidiprotare->temperatures( );
    for( GIDI::Styles::TemperatureInfos::const_iterator iter = temperatures.begin( ); iter != temperatures.end( ); ++iter ) {
      std::cout << "label = " << iter->heatedCrossSection( ) << "  temperature = " << iter->temperature( ).value( ) << std::endl;
      // fixme: check temp is supported by the data
      pt_assert_always(iter->temperature( ).value( ));
    }

    //FIXME: remove other temperatures
    temperatures.clear();
    temperatures.push_back(gidiprotare->temperatures( )[1]);
    double temperature_K = temperatures[1].temperature( ).value()*Unit::MeV / const_boltzmann; 

    MCGIDI::Transporting::MC MC ( *m_pops, gidiprotare->projectile( ).ID( ), &gidiprotare->styles( ), name, delay, 30.0 );
    // MC.setNuclearPlusCoulombInterferenceOnly( true );
    MC.sampleNonTransportingParticles( m_ctrdata.getGidiSampleNTP() );
    MC.set_ignoreENDF_MT5(true);
    MC.want_URR_probabilityTables(true);
    MC.setThrowOnError( false );
    // MC.setZeroDepositionIfAllProductsTracked
    // MC.setWantTerrellPromptNeutronDistribution(true);
   

    if( gidiprotare->protareType( ) != GIDI::ProtareType::single ) {
        PROMPT_THROW(CalcError, "ProtareType must be single");
    }
    int numberOfReactions = gidiprotare->numberOfReactions();
    std::cout <<"Isotope " << name << " has " << numberOfReactions << " reactions in total"<< "\n";


    std::set<int> reactionsToExclude;
    auto mcprotare = std::make_shared<MCGIDI::ProtareSingle>(*m_smr1, static_cast<GIDI::ProtareSingle const &>( *gidiprotare), *m_pops, MC, 
                                                                *m_particles, *m_domainHash, temperatures, reactionsToExclude );
    gidimodels.emplace_back(std::make_shared<GIDIModel>(const_neutron_pgd, name+"_all", mcprotare, temperature_K, bias, frac, elasticThreshold>0. ? elasticThreshold: minEKin, maxEKin));
    
    if(elasticThreshold>0.) {      
      std::set<int> nonElastic, elastic;
      for( int i = 0; i < numberOfReactions; ++i ) 
      {
        // The type of ENDF_MT can be found at https://t2.lanl.gov/nis/endf/mts.html
        if(gidiprotare->reaction(i)->ENDF_MT()==2)
        {
          elastic.emplace(i);
          break;
        }
      }

      
      auto mcProtare_nonelastic = std::make_shared<MCGIDI::ProtareSingle>(*m_smr1, static_cast<GIDI::ProtareSingle const &>( *gidiprotare), *m_pops, MC, 
                                                                  *m_particles, *m_domainHash, temperatures, elastic );
      gidimodels.emplace_back(std::make_shared<GIDIModel>(const_neutron_pgd, name+"_nonela", mcProtare_nonelastic, temperature_K, bias, frac, 0, elasticThreshold));
    }
    delete gidiprotare;
  }

  return std::move(gidimodels);
}



std::vector<std::shared_ptr<Prompt::GIDIModel>> Prompt::GIDIFactory::createPhotonGIDIModel(const std::vector<Prompt::IsotopeComposition> & vecComp, 
double bias, double minEKinElastic, double maxEKinElastic, double minEKinNonelastic, double maxEKinNonelastic) const
{
  std::vector<std::shared_ptr<GIDIModel>> gidimodels;
  MCGIDI::Vector<MCGIDI::Protare *> protares(vecComp.size());
  std::vector<std::tuple<std::shared_ptr<MCGIDI::ProtareSingle>, std::string, double, double>> singleProtares;


  // fixme: make shared pointer map to cache MCGIDI::ProtareSingle for repeated isotopes
  // the key should be the label (i.e. iter->heatedCrossSection( )) plus the  isotope name
  for(const auto& isotope : vecComp)
  {
    const std::string &name = isotope.name; //.substr(0,1);
    double frac = isotope.frac;
    
    if(!m_map->isProtareAvailable( PoPI::IDs::photon, name))
    {

      std::cout << "WARNING: photon data for "<< name << " are are not found. The cross section for isotope: " << isotope << "\" is ignored.\n";
      continue;
      // PROMPT_THROW2(DataLoadError, "GIDIFactory createPhotonGIDIModel failed to load data for " << name);
    }
    auto *gidiprotare =  (GIDI::Protare *) m_map->protare( *m_construction, *m_pops, "photon", name, "", "", true, true ) ;
    std::cout << "Using data file " << gidiprotare->realFileName( ) << std::endl;
    auto delay = GIDI::Transporting::DelayedNeutrons::on;
    if( !gidiprotare->isDelayedFissionNeutronComplete( ) ) 
    {
        std::cout << "WARNING: delayed neutron fission data for "<< name<< " are incomplete and are not included." << std::endl;
        delay = GIDI::Transporting::DelayedNeutrons::off;
        pt_assert_always(false);
    }
    else
      std::cout << "Delayed neutron fission data for "<< name<< " are included." << std::endl;

    GIDI::Transporting::Settings incompleteParticlesSetting( gidiprotare->projectile( ).ID( ),  delay);
    std::set<std::string> incompleteParticles;
    gidiprotare->incompleteParticles( incompleteParticlesSetting, incompleteParticles );
    std::cout << "# List of incomplete particles:";
    for( auto iter = incompleteParticles.begin( ); iter != incompleteParticles.end( ); ++iter ) {
        std::cout << " " << *iter;
    }
    std::cout << std::endl;
    
    GIDI::Styles::TemperatureInfos temperatures = gidiprotare->temperatures( );
    for( GIDI::Styles::TemperatureInfos::const_iterator iter = temperatures.begin( ); iter != temperatures.end( ); ++iter ) {
      std::cout << "label = " << iter->heatedCrossSection( ) << "  temperature = " << iter->temperature( ).value( ) << std::endl;
    }
   

    // std::string label( temperatures[0].heatedCrossSection( ) ); // fixme: heated? gridded
    std::string label( temperatures[0].griddedCrossSection( ) );

    double temperature_K = temperatures[0].temperature( ).value()*Unit::MeV / const_boltzmann; 

    MCGIDI::Transporting::MC MC ( *m_pops, gidiprotare->projectile( ).ID( ), &gidiprotare->styles( ), label, delay, 20.0 );
    // MC.setNuclearPlusCoulombInterferenceOnly( true );
    MC.sampleNonTransportingParticles( m_ctrdata.getGidiSampleNTP() );
    // MC.set_ignoreENDF_MT5(true);
    MC.want_URR_probabilityTables(false);
    MC.setThrowOnError( false );
   
    // if( gidiprotare->protareType( ) != GIDI::ProtareType::single ) {
    //     PROMPT_THROW(CalcError, "ProtareType must be single");
    // }
    
    std::set<int> exc;
    int numberOfReactions = gidiprotare->numberOfReactions();
    std::cout <<"Isotope " << name << " has " << numberOfReactions << " reactions in total"<< "\n";


    // if( gidiprotare.protareType( ) == GIDI::ProtareType::single ) 
    // auto mcProtare_nonelastic = std::make_shared<MCGIDI::ProtareSingle>(*m_smr1, static_cast<GIDI::ProtareSingle const &>( *gidiprotare), *m_pops, MC, 
    //                                                             *m_particles, *m_domainHash, temperatures, elastic );

    // else if( gidiprotare.protareType( ) == GIDI::ProtareType::composite ) {
    auto mcProtare_nonelastic = std::make_shared<MCGIDI::ProtareComposite> ( *m_smr1, static_cast<GIDI::ProtareComposite const &>( *gidiprotare ), *m_pops, MC, 
                                                                *m_particles, *m_domainHash, temperatures, exc );

    gidimodels.emplace_back(std::make_shared<GIDIModel>(const_photon_pgd, name, mcProtare_nonelastic, temperature_K, bias, frac, minEKinNonelastic, maxEKinNonelastic));

    delete gidiprotare;
  }


  return std::move(gidimodels);
}

