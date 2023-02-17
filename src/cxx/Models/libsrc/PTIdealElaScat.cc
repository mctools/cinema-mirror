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

#include <limits>

#include "PTIdealElaScat.hh"
#include "PTUnitSystem.hh"
#include "PTRandCanonical.hh"


Prompt::IdealElaScat::IdealElaScat(const std::string &cfgstring, double bias)
:Prompt::DiscreteModel(cfgstring, const_neutron_pgd,
                      std::numeric_limits<double>::min(), 10*Prompt::Unit::eV, bias)
{

}

Prompt::IdealElaScat::~IdealElaScat()
{
  std::cout<<"Destructing IdealElaScat physics " << m_modelName <<std::endl;
}


double Prompt::IdealElaScat::getCrossSection(double ekin) const
{
    return 1*Unit::barn*m_bias;
}

double Prompt::IdealElaScat::getCrossSection(double ekin, const Prompt::Vector &) const
{
  return getCrossSection(ekin);
}


void Prompt::IdealElaScat::generate(double ekin, const Prompt::Vector &dir, 
                        double &final_ekin, Prompt::Vector &final_dir) const
{
  final_ekin=ekin;

  //fixme: repeated code, first appared in the isotropicgun
  double r1 = m_rng.generate();
  double r2 = m_rng.generate();

  double u = 2*r1-1;
  double temp = sqrt(1-u*u);
  double v = temp*cos(2*M_PI*r2);
  double w = temp*sin(2*M_PI*r2);

  final_dir = Vector(u, v, w);
}
