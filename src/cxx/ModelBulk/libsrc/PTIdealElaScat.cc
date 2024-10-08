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

#include <limits>

#include "PTIdealElaScat.hh"
#include "PTUnitSystem.hh"
#include "PTRandCanonical.hh"


Prompt::IdealElaScat::IdealElaScat(double xs_barn, double density_per_aa3, double bias, double ekin_transfer)
:Prompt::DiscreteModel("IdealElaScat", const_neutron_pgd,
                      std::numeric_limits<double>::min(), std::numeric_limits<double>::max(), bias),
m_xs(xs_barn*Unit::barn*m_bias), m_density(density_per_aa3/Unit::Aa3), m_ekin_t(ekin_transfer*Unit::eV)
{
    std::cout<<"Created IdealElaScat physics. xs :" << xs_barn << ", bias: " 
             << bias << ", density_per_aa3 " << density_per_aa3 <<std::endl;

}

Prompt::IdealElaScat::~IdealElaScat()
{
  std::cout<<"Destructing IdealElaScat physics " << m_modelName <<std::endl;
}


double Prompt::IdealElaScat::getCrossSection(double ekin) const
{
  return m_xs;
}

double Prompt::IdealElaScat::getCrossSection(double ekin, const Prompt::Vector &) const
{
  return m_xs;
}


void Prompt::IdealElaScat::sampleReaction(double ekin, const Prompt::Vector &dir, 
                        double &final_ekin, Prompt::Vector &final_dir) const
{
  if (ekin<m_ekin_t)
  {
    final_ekin=ekin;
    // std::cout<<"EKIN: " << final_ekin << "<" << m_ekin_t << std::endl;
    final_dir.set(dir.x(), dir.y(), dir.z());
  }
  else
  {
    final_ekin=ekin-m_ekin_t;
    // std::cout<<"EKIN: " << final_ekin << "=" << ekin << "-" << m_ekin_t<< std::endl;

    //fixme: repeated code, first appared in the isotropicgun
    double r1 = m_rng.generate();
    double r2 = m_rng.generate();

    double u = 2*r1-1;
    double temp = sqrt(1-u*u);
    double v = temp*cos(2*M_PI*r2);
    double w = temp*sin(2*M_PI*r2);

    final_dir.set(u, v, w);
  }
}
