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

#include "PTScororNeutronSq.hh"

Prompt::ScororNeutronSq::ScororNeutronSq(const std::string &name, const Vector &samplePos, const Vector &refDir,
      double sourceSampleDist, double qmin, double qmax, unsigned numbin, bool kill, bool linear)
:Scoror1D("ScororNeutronSq_" + name, Scoror::ENTRY, std::make_unique<Hist1D>(qmin, qmax, numbin, linear)), m_samplePos(samplePos), m_refDir(refDir),
m_sourceSampleDist(sourceSampleDist),
m_kill(kill)
{}

Prompt::ScororNeutronSq::~ScororNeutronSq() {}

void Prompt::ScororNeutronSq::scoreLocal(const Vector &, double)
{
  PROMPT_THROW2(BadInput, m_name << " does not support scoreLocal()");
}


void Prompt::ScororNeutronSq::score(Prompt::Particle &particle)
{
  double angle_cos = particle.getDirection().angleCos(m_refDir);
  double dist = m_sourceSampleDist+(particle.getPosition()-m_samplePos).mag();
  double v = dist/particle.getTime();
  double ekin = 0.5*const_neutron_mass_evc2*v*v;
  //static approximation
  double q = neutronAngleCosine2Q(angle_cos, ekin, ekin);
  m_hist->fill(q, particle.getWeight());
  if(m_kill)
    particle.kill();
}

void Prompt::ScororNeutronSq::score(Prompt::Particle &particle, const DeltaParticle &dltpar)
{
  score(particle);
}
