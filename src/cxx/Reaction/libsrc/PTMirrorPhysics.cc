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

#include "PTMirrorPhysics.hh"
#include "PTUtils.hh"

Prompt::MirrorPhyiscs::MirrorPhyiscs(double mvalue, double weightCut)
:Prompt::BoundaryPhysics(Prompt::BoundaryPhysics::PhysicsType::Mirror), m_wcut(weightCut), m_wAtQ(0.)
{
  std::cout << "constructor mirror physics " << std::endl;
  //parameters sync with mcastas 2.7 guide component default value
  double m_i=mvalue;
  double R0=0.99;
  double Qc=0.0219;
  double alpha=6.07;
  double W=0.003;
  unsigned q_arr_size=1000;
  double q_max = 10;
  std::vector<double> q(q_arr_size);
  std::vector<double> r(q_arr_size);

  for(unsigned i=0;i<q_arr_size;i++)
  {
    q[i]=i*(q_max/q_arr_size);
    if(q[i]<Qc)
      r[i]=R0;
    else
      r[i]=0.5*R0*(1-tanh(( q[i]-m_i*Qc)/W))*(1-alpha*( q[i]-Qc));
  }
  m_table = std::make_shared<LookUpTable>(q, r, LookUpTable::kConst_Zero);
  std::cout << "constructor mirror physics completed" << std::endl;
}

void Prompt::MirrorPhyiscs::sampleFinalState(Prompt::Particle &particle) const
{
  double ekin = particle.getEKin();
  const auto &nDirInLab = particle.getDirection();

  Vector newDir = nDirInLab - m_refNorm*(2*(nDirInLab.dot(m_refNorm)));
  double angleCos = newDir.angleCos(nDirInLab);
  particle.setDirection(newDir);

  double Q = neutronAngleCosine2Q(angleCos, ekin, ekin); // elastic reflection
  m_wAtQ =  m_table->get(Q);

  // std::cout << "Q " << Q << " scale " << scaleWeight << std::endl;
  if(m_wcut > m_wAtQ)
  {
    if(m_wcut*m_rng.generate() < m_wAtQ )
    {
      m_wAtQ = m_wcut;
    }
    else
      particle.kill(Particle::KillType::BIAS);
  }
  particle.scaleWeight(m_wAtQ);

}
