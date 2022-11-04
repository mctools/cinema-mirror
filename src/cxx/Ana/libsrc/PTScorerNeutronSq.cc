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
      double sourceSampleDist, double qmin, double qmax, unsigned numbin, ScorerType stype, bool linear)
:Scorer1D("ScorerNeutronSq_" + name, stype, std::make_unique<Hist1D>("ScorerNeutronSq_" + name, qmin, qmax, numbin, linear)), m_samplePos(samplePos), m_refDir(refDir),
m_sourceSampleDist(sourceSampleDist), m_bwr(nullptr)
{
  if(stype==Scorer::ScorerType::ENTRY)
    m_kill=true;
  else if (stype==Scorer::ScorerType::ABSORB)
    m_kill=false;
  else
    PROMPT_THROW(BadInput, "ScorerNeutronSq can only be Scorer::ScorerType::ENTRY or Scorer::ScorerType::ABSORB");
  auto seed = Singleton<SingletonPTRand>::getInstance().getSeed();
  m_dataout.open("ScorerNeutronSq_" + name + "_seed"+std::to_string(seed)+".wgt");

  m_bwr = new BinaryWrite("ScorerNeutronSq_" + name + "_seed"+std::to_string(seed)+".record", false, true);
}

Prompt::ScorerNeutronSq::~ScorerNeutronSq()
{
  delete m_bwr;
  m_dataout.close();
}


void Prompt::ScorerNeutronSq::score(Prompt::Particle &particle)
{
  // fixme: the angle of cosine should be calculated as which line of code?
  // double angle_cos = (m_samplePos-particle.getPosition()).angleCos(m_refDir);
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
  << particle.getNumScat() << " "
  << particle.getWeight() <<  "\n";

  PromptRecord recode;
  recode.type = PromtRecodeType::SCRSQ;
  recode.sqRecode.ekin = particle.getEKin();
  recode.sqRecode.q = q;
  recode.sqRecode.qtrue = qtrue;
  recode.sqRecode.ekin_atbirth = particle.getEKin0();
  recode.sqRecode.ekin_tof = ekin;
  recode.sqRecode.time = tof_us;
  recode.sqRecode.weight = particle.getWeight();
  recode.sqRecode.scatNum = particle.getNumScat();
  recode.sqRecode.filldummy();
  m_bwr->record(recode);

  // static inline Vector& asVect( double (&v)[3] ) { return *reinterpret_cast<Vector*>(&v); }
  // static inline const Vector& asVect( const double (&v)[3] ) { return *reinterpret_cast<const Vector*>(&v); }

  // printf("Qe, Qtrue; Ekine , Ekin0 , Ekin; TOF; x y z\n");
  // printf("%f, %f, %.02e, %.02e, %.02e;  %.02f; %.02f %.02f %.02f\n\n", q, qtrue, ekin, particle.getEKin0(), particle.getEKin(),
  // tof_us, particle.getPosition().x(), particle.getPosition().y(), particle.getPosition().z());

  m_hist->fill(qtrue, particle.getWeight());
  if(m_kill)
    particle.kill(Particle::KillType::SCORE);
}
