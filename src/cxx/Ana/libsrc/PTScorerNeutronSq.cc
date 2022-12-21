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

#include "PTScorerNeutronSq.hh"
#include "PTRandCanonical.hh"

Prompt::ScorerNeutronSq::ScorerNeutronSq(const std::string &name, const Vector &samplePos, const Vector &refDir,
      double sourceSampleDist, double qmin, double qmax, unsigned numbin, ScorerType stype, bool qtrue, bool linear)
:Scorer1D("ScorerNeutronSq_" + name, stype, std::make_unique<Hist1D>("ScorerNeutronSq_" + name, qmin, qmax, numbin, linear)), m_samplePos(samplePos), m_refDir(refDir), 
m_sourceSampleDist(sourceSampleDist), m_qtrue(qtrue)
{
  if(stype==Scorer::ScorerType::ENTRY)
    m_kill=true;
  else if (stype==Scorer::ScorerType::ABSORB)
    m_kill=false;
  else
    PROMPT_THROW(BadInput, "ScorerNeutronSq can only be Scorer::ScorerType::ENTRY or Scorer::ScorerType::ABSORB");
}

Prompt::ScorerNeutronSq::~ScorerNeutronSq()
{
}


void Prompt::ScorerNeutronSq::score(Prompt::Particle &particle)
{
  if(particle.getPGD()!=2022)
    return; // for neutron only
    
  // bool m_qtrue=true; //fixme: this parameter should be defined in the constructor
  double angle_cos = particle.getDirection().angleCos(m_refDir);
  if(m_qtrue)
  {
    m_hist->fill(neutronAngleCosine2Q(angle_cos,  particle.getEKin0(), particle.getEKin()), particle.getWeight());
  }
  else //static approximation
  {
    // fixme: the angle of cosine should be calculated as which line of code?
    // double angle_cos = (m_samplePos-particle.getPosition()).angleCos(m_refDir);

    double dist = m_sourceSampleDist+(particle.getPosition()-m_samplePos).mag();
    double v = dist/particle.getTime();
    double ekin = 0.5*const_neutron_mass_evc2*v*v;
    double q = neutronAngleCosine2Q(angle_cos, ekin, ekin);
    m_hist->fill(q, particle.getWeight());
  }

  if(m_kill)
    particle.kill(Particle::KillType::SCORE);
}
