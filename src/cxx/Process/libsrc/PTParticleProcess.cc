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

#include "PTSurfaceProcess.hh"
#include "PTUnitSystem.hh" //const_neutron_pgd

#include <cmath>
#include <limits>
#include "NCrystal/NCrystal.hh"
#include "PTIdealElaScat.hh"
#include "PTPhysicsFactory.hh"
#include "PTActiveVolume.hh"
#include "PTNCrystalScat.hh"
#include "PTNCrystalAbs.hh"
#include "PTCfgParser.hh"
#include "PTMaterialDecomposer.hh"
#include "PTGIDI.hh"
#include "PTGIDIFactory.hh"
#include "PTCentralData.hh"
#include "PTStackManager.hh"

Prompt::ParticleProcess::ParticleProcess(const std::string &name, int pdg)
    : m_rng(Singleton<SingletonPTRand>::getInstance()),
      m_compModel(std::make_unique<CompoundModel>(pdg)),
      m_numdensity(0.), m_name(name)
{
  cfgPhysicsModel(name);
}

Prompt::ParticleProcess::~ParticleProcess() {}

double Prompt::ParticleProcess::macroCrossSection(const Prompt::Particle &particle) const
{
  double ekin = particle.hasEffEnergy() ? particle.getEffEKin() : particle.getEKin();
  const auto &dir = particle.hasEffEnergy() ? particle.getEffDirection() : particle.getDirection();
  return m_numdensity * m_compModel->totalCrossSection(ekin, dir);
}

void Prompt::ParticleProcess::sampleFinalState(Prompt::Particle &particle, double stepLength, bool hitWall) const
{
  if (m_compModel->getSupportedGPD() != particle.getPGD())
    PROMPT_THROW2(CalcError, "ParticleProcess::sampleFinalState " << m_name << " does not support particle " << particle.getPGD() << ", " << m_compModel->getSupportedGPD());

  if (!particle.isAlive())
    PROMPT_THROW(CalcError, "Particle is not alive");

  // if particle is escaped to the the next volume
  if (hitWall)
  {
    if (stepLength)
    {
      particle.scaleWeight(m_compModel->calculateWeight(stepLength * m_numdensity, true));
    }
    return;
  }

  // else
  double lab_ekin(0), comove_ekin(0);
  Vector lab_dir;

  if (particle.hasEffEnergy())
  {
    double ekineff = particle.getEffEKin();
    const auto &direff = particle.getEffDirection();

    Vector comove_dir;

    // sample in the comoving frame
    m_compModel->generate(ekineff, direff, comove_ekin, comove_dir);
    if (comove_ekin != -1) // non-capture fixme: this should not be called when EXITing
    {
      Vector v_comoving = comove_dir * std::sqrt(2 * comove_ekin / particle.getMass());
      // the rotatioal velocity
      auto v_rot = particle.getDirection() * particle.calcSpeed() - particle.getEffDirection() * particle.calcEffSpeed();

      particle.setEffEKin(comove_ekin);
      particle.setEffDirection(comove_dir);

      // bring back to the lab
      auto v_lab = v_comoving + v_rot;

      // set the final value in the lab frame
      double speed(0);
      v_lab.magdir(speed, lab_dir);
      // lab_ekin = particle.getEKin();
      lab_ekin = 0.5 * particle.getMass() * speed * speed;
    }
  }
  else
  {
    m_compModel->generate(particle.getEKin(), particle.getDirection(), lab_ekin, lab_dir);
  }

  // fixme: when a particle exiting a volume, a reaction channel is forced to sampled at the moment
  // lab_ekin could be -1 in those cases, but the transport keeps going, that is very confusing.
  // std::cout << particle.getEventID() << ", is alive? " << particle.isAlive() << ", wall? "<< hitWall<<
  //  ", labEkin " << lab_ekin   << ", effEkin " << particle.getEffEKin() << "\n";

  double weightCorrection = m_compModel->calculateWeight(stepLength * m_numdensity, false);

  // treating secondaries
  auto &stm = Singleton<StackManager>::getInstance();
  int secNum = stm.getUnweightedNum();
  for(int i=0; i<secNum; i++)
  {
    stm.scalceSecondary(i, weightCorrection);
  }

  // if it is an absorption reaction, the state of the particle is set,
  // but the energy and direction are kept for the subsequent capture scorers.
  if (lab_ekin == -1. || comove_ekin == -1.)
  {
    particle.kill(Particle::KillType::ABSORB);
  }
  else
  {
    particle.setEKin(lab_ekin);
    particle.setDirection(lab_dir);
  }
  particle.scaleWeight(weightCorrection);
}

double Prompt::ParticleProcess::sampleStepLength(const Prompt::Particle &particle) const
{
  if (m_compModel->getSupportedGPD() != particle.getPGD())
    PROMPT_THROW2(CalcError, "ParticleProcess::sampleStepLength " << m_name << " does not support particle " << particle.getPGD() << ", " << m_compModel->getSupportedGPD());

  double mxs = macroCrossSection(particle);
  if (mxs)
  {
    return -log(m_rng.generate()) / mxs;
  }
  else
  {
    return std::numeric_limits<double>::max();
  }
}

void Prompt::ParticleProcess::cfgPhysicsModel(const std::string &cfgstr)
{
  std::cout << "Configuring physics model: " << cfgstr << std::endl;
  pt_assert_always(!m_numdensity); //multiple configuration

  auto &pfact = Singleton<PhysicsFactory>::getInstance();
  PhysicsFactory::PhysicsType type = pfact.checkPhysicsType(cfgstr);

  auto &cd = Singleton<CentralData>::getInstance();

  const bool enablegidi = cd.getEnableGidi();
  const double gidimin = cd.getGidiThreshold();
  if (type == PhysicsFactory::PhysicsType::NC_RAW)
  {
    m_numdensity = pfact.nccalNumDensity(cfgstr);

    if(enablegidi)
    {
      std::cout << "enabled gidi model for " << cfgstr 
      << ", the switching energy at " << gidimin << "eV\n.";
      // NCrystal models
      if(gidimin>0)
         m_compModel->addPhysicsModel(std::make_shared<NCrystalScat>(cfgstr, 1.0, 0, gidimin));    
      
      // GIDI models 
      auto &nm = Prompt::Singleton<Prompt::MaterialDecomposer>::getInstance();
      auto isotopes = nm.decompose(cfgstr);
      for(const auto &v: isotopes)
        std::cout << v << std::endl;

      auto &gidifactory = Prompt::Singleton<Prompt::GIDIFactory>::getInstance();  
      auto models = gidifactory.createGIDIModel(isotopes, 1., gidimin<=0.?0:gidimin);

      for(const auto &v: models)
        m_compModel->addPhysicsModel(v);
    }
    else
    {
      m_compModel->addPhysicsModel(std::make_shared<NCrystalScat>(cfgstr, 1.0, 0));
      m_compModel->addPhysicsModel(std::make_shared<NCrystalAbs>(cfgstr, 1.0, 0));
    }

  }
  else if (type == PhysicsFactory::PhysicsType::NC_SCATTER)
  {
    CfgParser::StringCfg cfg = Singleton<CfgParser>::getInstance().parse(cfgstr);
    double scatter_bias = 1.0;
    cfg.getDoubleIfExist("scatter_bias", scatter_bias);

    double abs_bias = 1.0;
    cfg.getDoubleIfExist("abs_bias", abs_bias);

    std::string nccfg;
    cfg.getStringIfExist("nccfg", nccfg);
    m_numdensity = pfact.nccalNumDensity(nccfg);


    if(enablegidi)
    {
      std::cout << "enabled gidi model for " << cfgstr 
      << ", the switching energy at " << gidimin << "eV\n.";

      // GIDI models
      auto &nm = Prompt::Singleton<Prompt::MaterialDecomposer>::getInstance();
      auto isotopes = nm.decompose(nccfg);
      for(const auto &v: isotopes)
        std::cout << v << std::endl;


      auto &gidifactory = Prompt::Singleton<Prompt::GIDIFactory>::getInstance();  
      auto models = gidifactory.createGIDIModel(isotopes, abs_bias, gidimin<=0.?0:gidimin);

      for(const auto &v: models)
        m_compModel->addPhysicsModel(v);

      // NCrystal models 
      if(gidimin>0.)
      {
        m_compModel->addPhysicsModel(std::make_shared<NCrystalScat>(nccfg, scatter_bias, 0, gidimin));
      }
      else
        std::cout << "!The ncrystal elastic scatter is not created!\n";
    }
    else
    {
      m_compModel->addPhysicsModel(std::make_shared<NCrystalScat>(nccfg, scatter_bias, 0));
      m_compModel->addPhysicsModel(std::make_shared<NCrystalAbs>(nccfg, abs_bias, 0));
    }

  }
  else if (type == PhysicsFactory::PhysicsType::NC_IDEALSCAT)
  {
    auto idsct = pfact.createIdealElaScat(cfgstr);
    m_compModel->addPhysicsModel(idsct);
    m_numdensity =  reinterpret_cast<IdealElaScat *>(idsct.get())->getNumberDensity();
  }

}