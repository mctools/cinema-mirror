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
#include "PTNavManager.hh"

Prompt::MirrorPhyiscs::MirrorPhyiscs(double mvalue, double weightCut)
:Prompt::BoundaryPhysics(Prompt::BoundaryPhysics::PhysicsType::Mirror), m_wcut(weightCut), m_wAtQ(0.)
{
  std::cout << "constructor mirror physics " << std::endl;
  //parameters sync with mcastas 2.7 guide component default value
  m_m=mvalue;
  m_R0=0.99;
  m_Qc=0.0219;
  m_alpha=6.07;
  m_W=0.003;
  m_i_W=1/0.003;
}

void Prompt::MirrorPhyiscs::sampleFinalState(Prompt::Particle &particle) const
{
  auto &navMan = Singleton<NavManager>::getInstance();
  navMan.getNormal(particle.getPosition(), m_refNorm);

  double ekin = particle.getEKin();
  const auto &nDirInLab = particle.getDirection();

  Vector newDir = nDirInLab - m_refNorm*(2*(nDirInLab.dot(m_refNorm)));
  double angleCos = newDir.angleCos(nDirInLab);
  particle.setDirection(newDir);

  double Q = neutronAngleCosine2Q(angleCos, ekin, ekin); // elastic reflection
  m_wAtQ = Q<m_Qc ? m_R0 : 0.5*m_R0*(1-tanh(( Q-m_m*m_Qc)*m_i_W))*(1-m_alpha*( Q-m_Qc));

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
