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

#include "PTSurfaceProcess.hh"
#include "PTUnitSystem.hh" //const_neutron_pgd

#include <cmath>
#include <limits>
#include "NCrystal/NCrystal.hh"
#include "PTIdealElaScat.hh"
#include "PTPhysicsFactory.hh"
#include "PTActiveVolume.hh"

Prompt::BulkMaterialProcess::BulkMaterialProcess(const std::string &name)
    : m_rng(Singleton<SingletonPTRand>::getInstance()),
      m_compModel(std::make_unique<CompoundModel>(const_neutron_pgd)), // fixme:  neutron only for now, should be a dict later
      m_numdensity(0.), m_name(name)
{
  cfgPhysicsModel(name);
}

Prompt::BulkMaterialProcess::~BulkMaterialProcess() {}

double Prompt::BulkMaterialProcess::macroCrossSection(const Prompt::Particle &particle) const
{
  // if(m_compModel->containOriented())
  // {
  //   auto &activeVolume = Singleton<ActiveVolume>::getInstance();
  //
  //   std::cout << "global dir " << particle.getDirection() << ", pos " << particle.getPosition() << std::endl;
  //   std::cout << "local dir " << activeVolume.getTranslator().global2Local_direction(particle.getDirection())
  //             << ", pos " << activeVolume.getTranslator().global2Local(particle.getPosition()) << std::endl;
  //   std::cout << "local dir back " << activeVolume.getTranslator().local2Global_direction(particle.getDirection())
  //         << ", pos " << activeVolume.getTranslator().local2Global(particle.getPosition()) << std::endl;
  // }

  double ekin = particle.hasEffEnergy() ? particle.getEffEKin() : particle.getEKin();
  const auto &dir = particle.hasEffEnergy() ? particle.getEffDirection() : particle.getDirection();
  return m_numdensity * m_compModel->totalCrossSection(ekin, dir);
}

void Prompt::BulkMaterialProcess::sampleFinalState(Prompt::Particle &particle, double stepLength, bool hitWall) const
{
  if (m_compModel->getSupportedGPD() != particle.getPGD())
    PROMPT_THROW2(CalcError, "BulkMaterialProcess " << m_name << " does not support particle " << particle.getPGD() << " " << m_compModel->getSupportedGPD());

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
  particle.scaleWeight(m_compModel->calculateWeight(stepLength * m_numdensity, false));
}

double Prompt::BulkMaterialProcess::sampleStepLength(const Prompt::Particle &particle) const
{
  if (m_compModel->getSupportedGPD() != particle.getPGD())
    PROMPT_THROW2(CalcError, "BulkMaterialProcess " << m_name << " does not support particle " << particle.getPGD() << " " << m_compModel->getSupportedGPD());

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

void Prompt::BulkMaterialProcess::cfgPhysicsModel(const std::string &cfgstr)
{
  std::cout << "Configuring physics model: " << cfgstr << std::endl;
  pt_assert_always(!m_numdensity); //multiple configuration
  auto &pfact = Singleton<PhysicsFactory>::getInstance();
  PhysicsFactory::PhysicsType type = pfact.checkPhysicsType(cfgstr);

  if (type == PhysicsFactory::PhysicsType::NC_SCATTER)
  {
    std::cout << "PhysicsType type NC_SCATTER" << std::endl;
    m_compModel = pfact.createBulkMaterialProcess(cfgstr);
    // pt_assert_always(m_compModel->getModels().size() == 1);
    m_numdensity = pfact.nccalNumDensity(cfgstr); 
  }
  else if (type == PhysicsFactory::PhysicsType::NC_RAW)
  {
    std::cout << "PhysicsType type NC_RAW" << std::endl;
    m_compModel->addNCScaAbsModels(cfgstr, 1.0);
    m_numdensity = pfact.nccalNumDensity(cfgstr);
  }
  else if (type == PhysicsFactory::PhysicsType::NC_IDEALSCAT)
  {
    std::cout << "PhysicsType type NC_IDEALSCAT" << std::endl;
    m_compModel = pfact.createBulkMaterialProcess(cfgstr);
    // pt_assert_always(m_compModel->getModels().size() == 0);
    auto &aa = *reinterpret_cast<IdealElaScat *>(m_compModel->getModels()[0].get());
    m_numdensity = aa.getNumberDensity();
  }
}
