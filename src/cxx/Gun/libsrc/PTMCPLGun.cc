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

#include "PTMCPLGun.hh"
#include "kdsource.h"
#include <filesystem>

Prompt::MCPLGun::MCPLGun(const Particle &aParticle, const std::string &fn)
: PrimaryGun(aParticle), m_mode(std::filesystem::path(fn).extension()==".xml" ? 0 : 1)
{ 
  if(m_mode)
  {
    kdsource = KDS_open(fn.c_str());
  }
  else
  {
    m_mcplread = std::make_unique<MCPLParticleReader>(fn);
  }

}

Prompt::MCPLGun::~MCPLGun() { }

std::unique_ptr<Prompt::Particle> Prompt::MCPLGun::generate()
{
  if(m_mode)
  {
    char pt;
    mcpl_particle_t part;
    double w, v;
    int eof_reached;
    int use_kde = 1;
    double w_crit = 1;

    if(KDS_sample2(kdsource, m_particle, use_kde, w_crit, NULL, 1))
    {      
      PROMPT_THROW(CalcError, "eof_reached should not happen in the kde mode");
    }
  }
  else
  {
    if(m_mcplread->readOneParticle())
      PROMPT_THROW(CalcError, "failed to read particle from the mcpl file");
    m_particle = m_mcplread->getParticle();      
  }

  m_ekin = m_particle->ekin*Unit::MeV;
  m_ekin0 = m_ekin;

  m_pos.set(m_particle->position[0]*Unit::cm, 
            m_particle->position[1]*Unit::cm,
            m_particle->position[2]*Unit::cm);

  m_dir.set(m_particle->direction[0], 
             m_particle->direction[1],
             m_particle->direction[2]);
    
  m_eventid++;
  m_weight = m_particle->weight;
  m_alive=true;
  m_time = getTime();
  auto p = std::make_unique<Particle>(*this);
  return std::move(p);
}


void Prompt::MCPLGun::sampleEnergy(double &ekin)
{
  PROMPT_THROW(NotImplemented, "This method is forbidden in this class");
}

void Prompt::MCPLGun::samplePosDir(Vector &pos, Vector &dir) 
{
  PROMPT_THROW(NotImplemented, "This method is forbidden in this class");
}

double Prompt::MCPLGun::getParticleWeight()
{
  PROMPT_THROW(NotImplemented, "This method is forbidden in this class");
}

double Prompt::MCPLGun::getTime() 
{
  PROMPT_THROW(NotImplemented, "This method is forbidden in this class");
}



