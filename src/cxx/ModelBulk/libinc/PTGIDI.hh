#ifndef Prompt_GIDI_hh
#define Prompt_GIDI_hh

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

#include <string>
#include <memory>
#include <set>

#include "PromptCore.hh"
#include "PTDiscreteModel.hh"
#include "PTSingleton.hh"


namespace GIDI
{
  namespace Transporting
  {
    class Particles;
  }

  namespace Construction
  {
    class Settings;
  }
  
  namespace Map
  {
    class Map;
  }
}

namespace PoPI
{
  class Database;
}

namespace MCGIDI
{
  class DomainHash;
  class Protare;
  class URR_protareInfos;
  namespace Sampling
  {
    class StdVectorProductHandler;
  }
}  

namespace Prompt {
  class GIDIFactory;

  class GIDIModel  : public DiscreteModel {
  public:
    GIDIModel(const std::string &name, std::shared_ptr<MCGIDI::Protare> mcprotare, std::shared_ptr<MCGIDI::URR_protareInfos> urr_info, double bias=1.0);
    virtual ~GIDIModel();

    virtual double getCrossSection(double ekin) const override;
    virtual double getCrossSection(double ekin, const Vector &dir) const override;
    virtual void generate(double ekin, const Vector &dir, double &final_ekin, Vector &final_dir) const override;

  private:
    const GIDIFactory &m_factory;
    std::shared_ptr<MCGIDI::Protare> m_mcprotare;
    std::shared_ptr<MCGIDI::URR_protareInfos> m_urr_info;
    mutable MCGIDI::Sampling::StdVectorProductHandler *m_products;
    mutable double m_cacheEkin, m_cacheGidiXS;
  };

  class GIDIFactory {
  public:
    std::shared_ptr<GIDIModel> createGIDIModel(const std::string &name, double bias) const;
    int getHashID(double energy) const;
  private:
  
    friend class Singleton<GIDIFactory>;
    GIDIFactory();
    ~GIDIFactory();

    PoPI::Database *m_pops;
    GIDI::Map::Map *m_map;
    GIDI::Transporting::Particles *m_particles;
    GIDI::Construction::Settings *m_construction;
    MCGIDI::DomainHash *m_domainHash;
    std::set<int> m_reactionsToExclude;

  };
}

#endif
