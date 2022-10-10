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
#include "PTCfgParser.hh"

Prompt::ScorerNeutronSq::ScorerNeutronSq(const std::string &cfgstr)
{
  auto &ps = Singleton<CfgParser>::getInstance();
  // auto cfg = ps.getScorerCfg("Scorer=NeutronSq; name=SofQ;sample_position=0,0,1;beam_direction=0,0,1;src_sample_dist=30000;ScorerType=ENTRY;linear=true");
  auto cfg = ps.getScorerCfg(cfgstr);
  cfg.print();
  // auto samplePos = string2vec(words[2]);
  // auto neutronDir = string2vec(words[3]);
  // double moderator2SampleDist = ptstod(words[4]);
  // double minQ = ptstod(words[5]);
  // double maxQ = ptstod(words[6]);
  // int numBin = std::stoi(words[7]);
  // if(words[8]=="ABSORB")
  //   return std::make_shared<Prompt::ScorerNeutronSq>(words[1], samplePos, neutronDir, moderator2SampleDist, minQ, maxQ, numBin, Prompt::Scorer::ABSORB);
  // else if(words[8]=="ENTRY")
  //   return std::make_shared<Prompt::ScorerNeutronSq>(words[1], samplePos, neutronDir, moderator2SampleDist, minQ, maxQ, numBin, Prompt::Scorer::ENTRY);
  // else
  // {
  //   PROMPT_THROW2(BadInput, words[8] << " type is not supported by ScorerNeutronSq");
  //   return std::make_shared<Prompt::ScorerNeutronSq>(words[1], samplePos, neutronDir, moderator2SampleDist, minQ, maxQ, numBin, Prompt::Scorer::ENTRY);
  // }

}

Prompt::ScorerNeutronSq::ScorerNeutronSq(const std::string &name, const Vector &samplePos, const Vector &refDir,
  double sourceSampleDist, double qmin, double qmax, unsigned numbin,
  ScorerType stype, bool linear)
  :Scorer1D()
{
  init(name, samplePos, refDir, sourceSampleDist, qmin, qmax, numbin, stype, linear);
}

Prompt::ScorerNeutronSq::~ScorerNeutronSq()
{
  m_dataout.close();
}

void Prompt::ScorerNeutronSq::init(const std::string &name, const Vector &samplePos, const Vector &refDir,
  double sourceSampleDist, double qmin, double qmax, unsigned numbin, ScorerType stype, bool linear)
{
  // scorer
  m_name = "ScorerNeutronSq_" + name;
  m_type = stype;
  // scorer1D
  m_hist = std::make_unique<Hist1D>(qmin, qmax, numbin, linear);

  m_samplePos = samplePos;
  m_refDir = refDir;
  m_sourceSampleDist = sourceSampleDist;

  if(stype==Scorer::ENTRY)
    m_kill=true;
  else if (stype==Scorer::ABSORB)
    m_kill=false;
  else
    PROMPT_THROW(BadInput, "ScorerNeutronSq can only be Scorer::ENTRY or Scorer::ABSORB");
    auto seed = Singleton<SingletonPTRand>::getInstance().getSeed();
    m_dataout.open("ScorerNeutronSq_" + name + "_seed"+std::to_string(seed)+".wgt");
}

void Prompt::ScorerNeutronSq::score(Prompt::Particle &particle)
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
  << particle.getNumScat() << " "
  << particle.getWeight() <<  "\n";

  // printf("Qe, Qtrue; Ekine , Ekin0 , Ekin; TOF; x y z\n");
  // printf("%f, %f, %.02e, %.02e, %.02e;  %.02f; %.02f %.02f %.02f\n\n", q, qtrue, ekin, particle.getEKin0(), particle.getEKin(),
  // tof_us, particle.getPosition().x(), particle.getPosition().y(), particle.getPosition().z());

  m_hist->fill(qtrue, particle.getWeight());
  if(m_kill)
    particle.kill(Particle::SCORE);
}
