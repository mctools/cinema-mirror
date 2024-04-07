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
:Prompt::DiscreteModel(name+"_Gidi_"+std::to_string(mcprotare->numberOfReactions()), const_neutron_pgd,
                      lowerlimt,
                      upperlimt, 
                      bias),
m_factory(Prompt::Singleton<Prompt::GIDIFactory>::getInstance()), 
m_launcher(Singleton<Launcher>::getInstance()),
m_mcprotare(mcprotare), 
m_urr_info(nullptr),
m_products(new MCGIDI::Sampling::StdVectorProductHandler()),
m_cacheEkin(0.), 
m_cacheGidiXS(0.),
m_temperature(temperature),
m_frac(frac),
m_input(new MCGIDI::Sampling::Input(false, MCGIDI::Sampling::Upscatter::Model::B) )
{ 


  MCGIDI::Vector<MCGIDI::Protare *> protares(1);
  protares[0]= m_mcprotare.get();
  m_urr_info = new MCGIDI::URR_protareInfos(protares);


  m_input->m_temperature = const_boltzmann*temperature/Unit::keV;   // In keV/k;

    // std::cout << "!!!!!!!!! " << temperature_K << " " << const_boltzmann*293.15/Unit::MeV << std::endl;

  int numberOfReactions = m_mcprotare->numberOfReactions();
  std::cout <<"Model " << m_modelName << " has " << numberOfReactions << " reactions"<< "\n";
     
  for( int i = 0; i < numberOfReactions; ++i ) 
  {
    auto reaction =  m_mcprotare->reaction(i);
    // The type of ENDF_MT can be found at https://t2.lanl.gov/nis/endf/mts.html
    std::cout << "Reaction " << i << ", ENDF_MT=" << reaction->ENDF_MT()  << std::endl;
  }
}

Prompt::GIDIModel::~GIDIModel()
{
  delete m_products;
  delete m_urr_info;
  delete m_input;
  std::cout<<"Destructing GIDIModel " << m_modelName << " containing " << m_mcprotare->numberOfReactions() << " reactions" <<std::endl;
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

  const double ekin_MeV = ekin*1e-6;

  //fixme, do not care about temperature at the moment
  int hashIndex = m_factory.getHashID(ekin_MeV); //fixme, to remove if   pt_assert_always(ekin==m_cacheEkin)
  int reactionIndex = m_mcprotare->sampleReaction( *m_urr_info, hashIndex, m_input->m_temperature, ekin_MeV, m_cacheGidiXS, getRandNumber, nullptr );
  
  MCGIDI::Reaction const *reaction = m_mcprotare->reaction( reactionIndex );
  pt_assert_always(m_mcprotare->threshold( reactionIndex ) < ekin_MeV);

  m_products->clear();
  reaction->sampleProducts( m_mcprotare.get(), ekin_MeV, *m_input, getRandNumber, nullptr, *m_products );

  pt_assert_always(m_input->m_reaction==reaction);


  // // if MC.sampleNonTransportingParticles(true), many of the events are sampled in the centerOfMass
  // if(m_input->m_frame == GIDI::Frame::centerOfMass)
  // {
  //   if(m_input->m_sampledType == MCGIDI::Sampling::SampledType::firstTwoBody)
  //     std::cout << "MCGIDI::Sampling::SampledType::firstTwoBody\n";
  //   else if(m_input->m_sampledType == MCGIDI::Sampling::SampledType::secondTwoBody)
  //     std::cout << "MCGIDI::Sampling::SampledType::secondTwoBody\n";
  //   else if(m_input->m_sampledType == MCGIDI::Sampling::SampledType::uncorrelatedBody)
  //     std::cout << "MCGIDI::Sampling::SampledType::uncorrelatedBody\n";
  //   else if(m_input->m_sampledType == MCGIDI::Sampling::SampledType::unspecified)
  //     std::cout << "MCGIDI::Sampling::SampledType::unspecified\n";
  //   else if(m_input->m_sampledType == MCGIDI::Sampling::SampledType::photon)
  //     std::cout << "MCGIDI::Sampling::SampledType::photon\n";
  //   else
  //     PROMPT_THROW(CalcError, "unknown m_input->m_sampledType");

  //   // fixme
  //   // PROMPT_THROW(NotImplemented, "GIDI::Frame::centerOfMass product is not yet implemented");
  // }


  // debug and looking for the MT value for the selected reaction
  // std::cout << "ENDF MT" << reaction->ENDF_MT() <<  ", m_products->size() " <<  m_products->size() << std::endl;
  std::vector<Particle> secondaries;

  double totalekin = 0;
  double mu_cm = m_input->m_mu;
  double phi_cm = m_input->m_phi;

  for( std::size_t i = 0; i < m_products->size( ); ++i ) 
  {
    MCGIDI::Sampling::Product &aproduct = (*m_products)[i];

    if (aproduct.m_productIndex==11 || aproduct.m_productIndex==8) //neutron or gamma
    {
      Vector labdir;     
      double labekin(0);

      if(m_input->m_frame == GIDI::Frame::centerOfMass)
      {
        // recepice from https://docs.openmc.org/en/stable/methods/neutron_physics.html#transforming-a-particle-s-coordinates
        // also see P129, eq. 6.5, ENDF-6 Formats Manual CSEWG Document ENDF-102, Report BNL-90365-2009 Rev.1
        // work around for 0,0,1 direction case, where mu_wfactor is infinity   
        Vector swapdir(dir);
        if(dir.z()==1.)
        {
          std::swap(swapdir.x(), swapdir.z());
        }

        double A =  aproduct.m_productMass/m_input->m_projectileMass;
        double Ap1 = A+1;
        double i_Ap1 = 1./Ap1;

        // dir, labdir
        double labekin_MeV = aproduct.m_kineticEnergy + (ekin_MeV + 2*mu_cm*Ap1*sqrt(ekin_MeV*aproduct.m_kineticEnergy))*i_Ap1*i_Ap1;
        labekin = labekin_MeV*1e6;
        pt_assert_always(labekin_MeV);

        double i_labekin_MeV = 1./labekin_MeV;
        double mu_lab = m_input->m_mu*sqrt(aproduct.m_kineticEnergy*i_labekin_MeV) + i_Ap1*sqrt(ekin_MeV*i_labekin_MeV);

        double cosphi_com = cos(m_input->m_phi);
        double sinphi_com = sin(m_input->m_phi);
        double mufactor = sqrt(1-mu_lab*mu_lab);
        double wfactor = sqrt(1-swapdir.z()*swapdir.z());
        double mu_wfactor = mufactor/wfactor;
        
        Vector dircm(aproduct.m_px_vx, aproduct.m_py_vy, aproduct.m_pz_vz);
        dircm.setMag(1);

        // std::cout << "center of mass dir " << dircm << "\n";

        labdir.set((mu_lab*swapdir.x() +  mu_wfactor*( swapdir.x()*swapdir.z()*cosphi_com - swapdir.y()*sinphi_com)),
                   (mu_lab*swapdir.y() +  mu_wfactor*( swapdir.y()*swapdir.z()*cosphi_com - swapdir.x()*sinphi_com)),
                   (mu_lab*swapdir.z() -  mufactor * wfactor * cosphi_com ));

        if(dir.z()==1.)
        {
          std::swap(labdir.x(), labdir.z());
          std::swap(labdir.x(), labdir.y());
        }

        // std::cout << "cm " << aproduct.m_kineticEnergy << ", lab " << labekin_MeV << ", ratio " << aproduct.m_kineticEnergy/labekin_MeV 
        //           <<", lab direction mag " << labdir.mag() << " "  << dir << " " << labdir <<"\n";
      }
      else
      {
        labekin = aproduct.m_kineticEnergy*1e6;
        labdir.set(aproduct.m_px_vx, aproduct.m_py_vy, aproduct.m_pz_vz);
        labdir.setMag(1.);
      }

      totalekin += labekin; // accumulate the energy that will be carried by a particle in the later simulation. 

      // make secondary
      Particle primary; 
      m_launcher.copyCurrentParticle(primary);

      primary.setEKin(labekin);
      primary.setDirection(labdir);
      primary.setTime(primary.getTime() + aproduct.m_birthTimeSec);

      if(aproduct.m_productIndex==11) // fixme: only neutron to the stack for now
        secondaries.push_back(primary);
    }
    
  }

  // All secondary particles that are not simulated by prompt are contributed to the "energy deposition".
  // So, in the case that neutron is the only transporting particle, the energy deposition is calculated as incident neutorn kinetic energy 
  // plus Q and substrcut the total kinetic energy of all the tracking particles. 

  // printf("MT%d, deposition %e\n\n", reaction->ENDF_MT(), ekin+reaction->finalQ(ekin_MeV)*1e6-totalekin);


  // Neutron die as absorption
  if(secondaries.size()==0)
  {
    // essentially killing the current active particle in the launcher
    final_ekin=ENERGYTOKEN_ABSORB;
  }
  else if(secondaries.size()==1)
  {
    // essentially modifying the current active particle in the launcher
    //fixme: how about a reaction produce only one delayed particle? there is no way to treat the time in this function
    final_ekin = secondaries[0].getEKin();
    final_dir = secondaries[0].getDirection();
  }
  else
  {
    // essentially killing the current active particle in the launcher
    final_ekin=ENERGYTOKEN_ABSORB;

    for(const auto &p: secondaries)
    {
      Singleton<StackManager>::getInstance().addSecondary(p, m_factory.getCentralData().getEnableGidiPowerIteration());
    }
    // std::cout << p << std::endl;
    // //Fixme: mcpl writer will be used to record neutron multiplication for keff application
    // PROMPT_THROW(NotImplemented, "neutron multiplication is not yet supported");
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

std::vector<std::shared_ptr<Prompt::GIDIModel>> Prompt::GIDIFactory::createGIDIModel(std::vector<Prompt::IsotopeComposition> vecComp, 
double bias, double minEKinElastic, double maxEKinElastic, double minEKinNonelastic, double maxEKinNonelastic) const
{
  std::vector<std::shared_ptr<GIDIModel>> gidimodels;
  MCGIDI::Vector<MCGIDI::Protare *> protares(vecComp.size());
  std::vector<std::tuple<std::shared_ptr<MCGIDI::ProtareSingle>, std::string, double, double>> singleProtares;


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
    }

    std::string label( temperatures[0].heatedCrossSection( ) );
    double temperature_K = temperatures[0].temperature( ).value() * Unit::MeV / const_boltzmann; 

    MCGIDI::Transporting::MC MC ( *m_pops, gidiprotare->projectile( ).ID( ), &gidiprotare->styles( ), label, delay, 20.0 );
    // MC.setNuclearPlusCoulombInterferenceOnly( false );
    MC.sampleNonTransportingParticles( m_ctrdata.getGidiSampleNTP() );
    // MC.set_ignoreENDF_MT5(true);

   

    if( gidiprotare->protareType( ) != GIDI::ProtareType::single ) {
        PROMPT_THROW(CalcError, "ProtareType must be single");
    }
    
    std::set<int> nonElastic, elastic;
    int numberOfReactions = gidiprotare->numberOfReactions();
    std::cout <<"Isotope " << name << " has " << numberOfReactions << " reactions in total"<< "\n";
      
    for( int i = 0; i < numberOfReactions; ++i ) 
    {
      // The type of ENDF_MT can be found at https://t2.lanl.gov/nis/endf/mts.html
      if(gidiprotare->reaction(i)->ENDF_MT()==2)
      {
        elastic.emplace(i);
      }
      else
      {
        nonElastic.emplace(i);
      }
    }

    auto mcProtare_elastic = std::make_shared<MCGIDI::ProtareSingle>(*m_smr1, static_cast<GIDI::ProtareSingle const &>( *gidiprotare), *m_pops, MC, 
                                                             *m_particles, *m_domainHash, temperatures, nonElastic );

    auto mcProtare_nonelastic = std::make_shared<MCGIDI::ProtareSingle>(*m_smr1, static_cast<GIDI::ProtareSingle const &>( *gidiprotare), *m_pops, MC, 
                                                                *m_particles, *m_domainHash, temperatures, elastic );

    gidimodels.emplace_back(std::make_shared<GIDIModel>(name, mcProtare_elastic,    temperature_K, bias, frac, minEKinElastic, maxEKinElastic));
    gidimodels.emplace_back(std::make_shared<GIDIModel>(name, mcProtare_nonelastic, temperature_K, bias, frac, minEKinNonelastic, maxEKinNonelastic));

    delete gidiprotare;
  }

  // auto URR_protare_infos = std::make_shared<MCGIDI::URR_protareInfos>();
  // for(auto s : singleProtares)
  //   gidimodels.emplace_back(std::make_shared<GIDIModel>(std::get<1>(s), std::get<0>(s), std::get<2>(s), bias, std::get<3>(s), minEKin, maxEKin));

  return std::move(gidimodels);
}
