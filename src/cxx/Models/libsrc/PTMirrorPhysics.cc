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
:Prompt::DiscreteModel("MirrorPhysics", const_neutron_pgd, std::numeric_limits<double>::min(), 10*Prompt::Unit::eV),
m_wcut(weightCut), m_rng(Singleton<SingletonPTRand>::getInstance())
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


Prompt::MirrorPhyiscs::~MirrorPhyiscs()
{}

double Prompt::MirrorPhyiscs::getCrossSection(double ekin) const
{
  return 0.;
}

double Prompt::MirrorPhyiscs::getCrossSection(double ekin, const Vector &dir) const
{
  return 0.;
}

void Prompt::MirrorPhyiscs::generate(double ekin, const Vector &nDirInLab, double &final_ekin, Vector &reflectionNor, double &scaleWeight) const
{
  final_ekin=ekin;
  reflectionNor = nDirInLab - reflectionNor*(2*(nDirInLab.dot(reflectionNor)));
  double angleCos = reflectionNor.angleCos(nDirInLab);

  double Q = neutronAngleCosine2Q(angleCos, ekin, ekin);
  scaleWeight =  m_table->get(Q);
  // std::cout << "Q " << Q << " scale " << scaleWeight << std::endl;
  if(m_wcut > scaleWeight)
  {
    if(m_wcut*m_rng.generate() < scaleWeight )
    {
      scaleWeight = m_wcut;
    }
    else
      final_ekin = -1.0; //paprose kill
  }


}
