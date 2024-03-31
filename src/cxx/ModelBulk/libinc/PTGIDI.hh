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
#include "PTCentralData.hh"
#include "PTLauncher.hh"


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

namespace LUPI
{
  class StatusMessageReporting;
}

namespace MCGIDI
{
  class DomainHash;
  class Protare;
  class URR_protareInfos;
  namespace Sampling
  {
    class StdVectorProductHandler;
    class Input;
  }
}  

namespace Prompt {
  class GIDIFactory;

  class GIDIModel  : public DiscreteModel {
  public:
    GIDIModel(const std::string &name, std::shared_ptr<MCGIDI::Protare> mcprotare,
              double temperature, double bias=1.0, double frac=1.0, double lowerlimt = 0., double upperlimt = std::numeric_limits<double>::max());
    virtual ~GIDIModel();

    virtual double getCrossSection(double ekin) const override;
    virtual double getCrossSection(double ekin, const Vector &dir) const override;
    virtual void generate(double ekin, const Vector &dir, double &final_ekin, Vector &final_dir) const override;

  private:
    const GIDIFactory &m_factory;
    Launcher &m_launcher; 
    std::shared_ptr<MCGIDI::Protare> m_mcprotare;
    MCGIDI::URR_protareInfos *m_urr_info;
    mutable MCGIDI::Sampling::StdVectorProductHandler *m_products;
    mutable double m_cacheEkin, m_cacheGidiXS;
    const double m_temperature, m_frac;
    MCGIDI::Sampling::Input *m_input;
  };

  

  class IsotopeComposition;

  class GIDIFactory {
  public:
    std::vector<std::shared_ptr<GIDIModel>> createGIDIModel(std::vector<IsotopeComposition> iso,  double bias=1. , 
                            double minEKinElastic=0, double maxEKinElastic=std::numeric_limits<double>::max(),
                            double minEKinNonelastic=0, double maxEKinNonelastic=std::numeric_limits<double>::max()) const;

    int getHashID(double energy) const;
    bool available() const;
    CentralData &getCentralData() const {return m_ctrdata;};

    inline bool NCrystal4Elastic(double ekin) const 
    {
      return ekin < m_ctrdata.getGidiThreshold();
    };


  private:
  
    friend class Singleton<GIDIFactory>;
    GIDIFactory();
    ~GIDIFactory();

    CentralData &m_ctrdata;
    PoPI::Database *m_pops;
    GIDI::Map::Map *m_map;
    GIDI::Transporting::Particles *m_particles;
    GIDI::Construction::Settings *m_construction;
    MCGIDI::DomainHash *m_domainHash;
    LUPI::StatusMessageReporting *m_smr1;

  };
}

#endif
