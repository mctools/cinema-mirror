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

#include "PTCompoundModel.hh"
#include "PTNCrystalScat.hh"
#include "PTNCrystalAbs.hh"
#include "PTPhysicsModel.hh"

Prompt::CompoundModel::CompoundModel()
:m_cache({}), m_oriented(false), m_rng( Singleton<SingletonPTRand>::getInstance() )
{}

Prompt::CompoundModel::~CompoundModel() {}

void Prompt::CompoundModel::addPhysicsModel(const std::string &cfg, double bias)
{
  if(bias!=1.)
    std::cout << "material " << cfg << " has a bias of " << bias << std::endl;

  // fixme: absoption should be seperated once ENDF data model is available
  m_models.emplace_back(std::make_shared<NCrystalAbs>(cfg, bias));
  // cache_xs and bias will be updated once a calculation is required.
  // so the initial value can be arbitrary.
  m_cache.cache_xs.push_back(0.);
  m_cache.bias.push_back(1.);
  if(m_models.back()->isOriented())
    m_oriented=true;

  m_models.emplace_back(std::make_shared<NCrystalScat>(cfg, bias));
  m_cache.cache_xs.push_back(0.);
  m_cache.bias.push_back(1.);
  if(m_models.back()->isOriented())
    m_oriented=true;
}


double Prompt::CompoundModel::totalCrossSection(double ekin, const Vector &dir) const
{
  if(sameInquiryAsLastTime(ekin, dir))
  {
    return m_cache.tot;
  }
  else
  {
    double xs(0.);
    for(unsigned i=0;i<m_models.size();i++)
    {
      double channelxs = m_oriented ? m_models[i]->getCrossSection(ekin, dir) :
                                      m_models[i]->getCrossSection(ekin);
      m_cache.cache_xs[i] = channelxs;
      m_cache.bias[i] = m_models[i]->getBias();
      xs += channelxs;
    }
    m_cache.tot = xs;
    m_cache.ekin = ekin;
    m_cache.dir = dir;
    return xs;
  }
}

void Prompt::CompoundModel::sample(double ekin, const Vector &dir, double &final_ekin, Vector &final_dir, double &scaleWeight) const
{
  if(!sameInquiryAsLastTime(ekin, dir))
    printf("WARNING, sampling event with different incident energy and/or direction\n");

  //if xs is zero, do nothing
  if(!m_cache.tot)
  {
    final_ekin = ekin;
    final_dir = dir;
    return;
  }

  double r1 =  m_rng.generate();
  unsigned i=0;
  double p(0.), i_tot(1./m_cache.tot);
  for(; i<m_cache.cache_xs.size(); i++) //fixme: this is only faster when the number of physics model is small
  {
    p += m_cache.cache_xs[i]*i_tot;
    if(p > r1)
      break;
  }
  m_models[i]->generate(ekin, dir, final_ekin, final_dir, scaleWeight);
  m_cache.selectedBias = m_models[i]->getBias();
  // std::cout << "selected model " << m_models[i]->getName() << " "
  // << " total model num " << m_models.size() << std::endl;
}

//call it right after cross section is updated
double Prompt::CompoundModel::calculateWeight(double lengthRho, bool selBiase)
{
  double factor(1.);
  for(size_t i=0;i<m_models.size();i++)
  {
    double modbias(m_models[i]->getBias());
    if (modbias==1.) continue;
    factor *= exp( (m_cache.bias[i]-1.)*lengthRho* m_cache.cache_xs[i]/m_cache.bias[i] );
    // std::cout << "exponet " <<  (m_cache.bias[i]-1.)*lengthRho* m_cache.cache_xs[i] <<
    // " "<< exp( (m_cache.bias[i]-1.)*lengthRho* m_cache.cache_xs[i] ) << std::endl;
  }
  // std::cout << "selectedBias " << m_cache.selectedBias << " factor " << factor
  // << std::endl;
  return (m_cache.selectedBias!=1. && selBiase ) ?  (factor/m_cache.selectedBias) : factor;
}
