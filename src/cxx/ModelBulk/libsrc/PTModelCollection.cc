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

#include "PTModelCollection.hh"
#include "PTNCrystalScat.hh"
#include "PTNCrystalAbs.hh"
#include "PTPhysicsModel.hh"
#include "PTActiveVolume.hh"

Prompt::ModelCollection::ModelCollection(int gpd)
:m_cache({}), m_containsOriented(false), m_rng( Singleton<SingletonPTRand>::getInstance() ),
 m_forgpd(gpd)
{}

Prompt::ModelCollection::~ModelCollection() {}


void Prompt::ModelCollection::addPhysicsModel(std::shared_ptr<Prompt::DiscreteModel> model)
{
  if(!model->getModelValidity().rightParticleType(m_forgpd))
    PROMPT_THROW2(BadInput, "the model is not aimed for suitable for particle GPD " << m_forgpd);
  m_models.emplace_back(model);

  // cache_xs and bias will be updated once a calculation is required.
  // so the initial value can be arbitrary.
  m_cache.cache_xs.push_back(0.);
  m_cache.bias.push_back(1.); // to be update in totalCrossSection
  if(m_models.back()->isOriented())
    m_containsOriented=true;
}



double Prompt::ModelCollection::totalCrossSection(double ekin, const Vector &dir) const
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
      double channelxs(0);
      if(m_models[i]->isOriented())
      {
        auto &activeVolume = Singleton<ActiveVolume>::getInstance();
        m_localdir =  activeVolume.getGeoTranslator().global2Local_direction(dir);
        channelxs = m_models[i]->getCrossSection(ekin, m_localdir) ;
      }
      else  {
        channelxs = m_models[i]->getCrossSection(ekin);
        // std::cout << "model name: " << m_models[i]->getName()
        // << ", ekin=" << ekin
        // << ", biasing=" << m_models[i]->getBias() << ", channelxs=" << channelxs << "\n";
      }
      m_cache.cache_xs[i] = channelxs;
      m_cache.bias[i] = m_models[i]->getBias();
      xs += channelxs;
    }
    // std::cout << "total xs " << xs << "\n\n";

    m_cache.tot = xs;
    m_cache.ekin = ekin;
    m_cache.dir = dir;
    return xs;
  }
}

void Prompt::ModelCollection::generate(double ekin, const Vector &dir, double &final_ekin, Vector &final_dir) const
{
  if(!sameInquiryAsLastTime(ekin, dir))
  {
    //fixme:!!
    printf("WARNING, sampling event with different incident energy and/or direction\n");
    final_ekin = ekin;
    final_dir = dir;
    return;
  }

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

  if(m_models[i]->isOriented())
  {
    auto &activeVolume = Singleton<ActiveVolume>::getInstance();
    m_models[i]->generate(ekin, m_localdir, final_ekin, final_dir);
    final_dir =  activeVolume.getGeoTranslator().local2Global_direction(final_dir);
  }
  else
    m_models[i]->generate(ekin, dir, final_ekin, final_dir);
    
  m_cache.selectedBias = m_models[i]->getBias();
  // std::cout << "selected model " << m_models[i]->getName() 
  // << ", biasing " << m_cache.selectedBias
  // << ", total model num " << m_models.size() << std::endl;

}

//this shoule be called right after cross section is updated
double Prompt::ModelCollection::calculateWeight(double lengthRho, bool hitWall)
{
  double factor(1.);
  for(size_t i=0;i<m_models.size();i++)
  {
    double modbias(m_models[i]->getBias());
    if (modbias==1.) continue;
    //this factor contributed in this step of lengthRho by this model
    factor *= exp( (m_cache.bias[i]-1.)*lengthRho* m_cache.cache_xs[i]/m_cache.bias[i] );
  }
  // std::cout << "selectedBias " << m_cache.selectedBias << " factor " << factor
  // << std::endl;
  // assert(m_cache.selectedBias);
  return (m_cache.selectedBias!=0 && m_cache.selectedBias!=1. && !hitWall ) ?  (factor/m_cache.selectedBias) : factor;
}
