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

#include "PTBulkPhysics.hh"
#include <cmath>
#include <limits>
#include "NCrystal/NCrystal.hh"

Prompt::BulkPhysics::BulkPhysics()
:m_rng(Singleton<SingletonPTRand>::getInstance()),
m_compModel(std::make_unique<CompoundModel>()),
m_numdensity(0.) { }

Prompt::BulkPhysics::~BulkPhysics() { }

double Prompt::BulkPhysics::macroCrossSection(const Prompt::Particle &particle) const
{
  double ekin = particle.hasEffEnergy() ? particle.getEffEKin() : particle.getEKin();
  const auto &dir = particle.hasEffEnergy() ? particle.getEffDirection() : particle.getDirection();
  return m_numdensity*m_compModel->totalCrossSection(ekin, dir);
}

void Prompt::BulkPhysics::sampleFinalState(Prompt::Particle &particle, double stepLength, bool hitWall) const
{
  if(!particle.isAlive())
    return;

  double lab_ekin;
  Vector lab_dir;

  if(particle.hasEffEnergy())
  // if(false)
  {
    double ekineff =  particle.getEffEKin();
    const auto &direff = particle.getEffDirection();

    double comove_ekin;
    Vector comove_dir;

    // SURE CHANGE REAL AND EFF both

    // sample in the comoving frame
    m_compModel->generate(ekineff, direff, comove_ekin, comove_dir);
    if(lab_ekin!=-1) //non-capture fixme: this should not be called when EXITing
    {
      Vector v_comoving = comove_dir*std::sqrt(2*comove_ekin/particle.getMass());

      // the rotatioal velocity
      auto v_rot = particle.getDirection()*particle.calcSpeed()-particle.getEffDirection()*particle.calcEffSpeed();

      // bring back to the lab
      auto v_final = v_comoving + v_rot;

      // set the final value in the lab frame
      double speed(0);
      v_final.magdir(speed, lab_dir);
      // lab_ekin = particle.getEKin();
      lab_ekin = 0.5*particle.getMass()*speed*speed;
      // std::cout << particle.getEKin() << " " << lab_dir << " "<<  particle.getEffEKin()  << " " << lab_ekin << "\n";
    }
  }
  else
  {
    m_compModel->generate(particle.getEKin(), particle.getDirection(), lab_ekin, lab_dir);
  }

  if(!hitWall)
  {

    particle.setEKin(lab_ekin);
    particle.setDirection(lab_dir);
    if(lab_ekin==-1.) // fixme: are we sure all -1 means capture??
    {
      particle.kill(Particle::KillType::ABSORB);
    }
  }

  if(stepLength)
    particle.scaleWeight(m_compModel->calculateWeight(stepLength*m_numdensity, hitWall));
}


double Prompt::BulkPhysics::sampleStepLength(const Prompt::Particle &particle) const
{
  double mxs = macroCrossSection(particle);
  if(mxs)
  {
    return -log(m_rng.generate())/mxs;
  }
  else
  {
    return std::numeric_limits<double>::max();
  }
}


double Prompt::BulkPhysics::calNumDensity(const std::string &cfg)
{
  NCrystal::MatCfg matcfg(cfg);
  auto info = NCrystal::createInfo(matcfg);
  if(info->hasNumberDensity())
    return info->getNumberDensity().get() / Unit::Aa3;
  else
  {
    PROMPT_THROW2(CalcError, "material has no number density " << cfg);
    return 0.;
  }
}

void Prompt::BulkPhysics::setComposition(const std::string &cfg, double bias)
{
  assert(!m_numdensity);
  m_compModel->addPhysicsModel(cfg, bias);
  m_numdensity = calNumDensity(cfg);
}
