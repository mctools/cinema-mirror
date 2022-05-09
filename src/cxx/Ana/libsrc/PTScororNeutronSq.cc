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
#include "PTRandCanonical.hh"

Prompt::ScororNeutronSq::ScororNeutronSq(const std::string &name, const Vector &samplePos, const Vector &refDir,
      double sourceSampleDist, double qmin, double qmax, unsigned numbin, ScororType stype, bool linear)
:Scoror1D("ScororNeutronSq_" + name, stype, std::make_unique<Hist1D>(qmin, qmax, numbin, linear)), m_samplePos(samplePos), m_refDir(refDir),
m_sourceSampleDist(sourceSampleDist)
{
  if(stype==Scoror::ENTRY)
    m_kill=true;
  else if (stype==Scoror::ABSORB)
    m_kill=false;
  else
    PROMPT_THROW(BadInput, "ScororNeutronSq can only be Scoror::ENTRY or Scoror::ABSORB");
    auto seed = Singleton<SingletonPTRand>::getInstance().getSeed();
    m_dataout.open("ScororNeutronSq_" + name + "_seed"+std::to_string(seed)+".wgt");
}

Prompt::ScororNeutronSq::~ScororNeutronSq()
{
  m_dataout.close();
}


void Prompt::ScororNeutronSq::score(Prompt::Particle &particle)
{
  double angle_cos = particle.getDirection().angleCos(m_refDir);
  double dist = m_sourceSampleDist+(particle.getPosition()-m_samplePos).mag();
  double v = dist/particle.getTime();
  double ekin = 0.5*const_neutron_mass_evc2*v*v;
  //static approximation
  double q = neutronAngleCosine2Q(angle_cos, ekin, ekin);

  double qtrue = neutronAngleCosine2Q(angle_cos,  particle.getEKin0(), particle.getEKin());
  double tof_us = particle.getTime()*1e6;
  m_dataout << tof_us << " "
  <<  particle.getPosition().x() << " "
  <<  particle.getPosition().y() << " "
  <<  particle.getPosition().z() << " "
  << q << " "
  << qtrue << " "
  << ekin << " "
  << particle.getEKin0() << " "
  << particle.getEKin() << " "
  << particle.getWeight() <<  "\n";

  // printf("Qe, Qtrue; Ekine , Ekin0 , Ekin; TOF; x y z\n");
  // printf("%f, %f, %.02e, %.02e, %.02e;  %.02f; %.02f %.02f %.02f\n\n", q, qtrue, ekin, particle.getEKin0(), particle.getEKin(),
  // tof_us, particle.getPosition().x(), particle.getPosition().y(), particle.getPosition().z());

  m_hist->fill(q, particle.getWeight());
  if(m_kill)
    particle.kill(Particle::SCORE);
}
