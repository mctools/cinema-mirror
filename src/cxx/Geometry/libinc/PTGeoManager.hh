#ifndef Prompt_GeoManager_hh
#define Prompt_GeoManager_hh

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

#include <string>
#include <map>
#include <unordered_map>
#include "PromptCore.hh"
#include "PTMaterialPhysics.hh"
#include "PTSingleton.hh"
#include "PTScorer.hh"
#include "PTPrimaryGun.hh"
#include "PTPhysicsModel.hh"

namespace Prompt {

  struct VolumePhysicsScorer { // to attach to a volume
    std::shared_ptr<MaterialPhysics> bulkPhysics; //bulk physics
    std::shared_ptr<PhysicsModel> boundaryPhysics; //boundary physics
    std::vector< std::shared_ptr<Scorer> >  scorers; /*scorer name, scorer*/

    std::vector< std::shared_ptr<Scorer> >  surface_scorers;
    std::vector< std::shared_ptr<Scorer> >  entry_scorers;
    std::vector< std::shared_ptr<Scorer> >  propagate_scorers;
    std::vector< std::shared_ptr<Scorer> >  exit_scorers;
    std::vector< std::shared_ptr<Scorer> >  absorb_scorers;

    void sortScorers();
  };

  using VPSMap = std::unordered_map<size_t, std::shared_ptr<VolumePhysicsScorer>>;

  class GeoManager  {
  public:
    void loadFile(const std::string &loadFile);
    std::shared_ptr<MaterialPhysics> getMaterialPhysics(const std::string &name);
    std::shared_ptr<Scorer> getScorer(const std::string &name);
    size_t numMaterialPhysics() {return m_globelPhysics.size();}
    size_t numScorer() {return m_globelScorers.size();}
    std::string getLogicalVolumeScorerName(unsigned logid);
    const std::string &getLogicalVolumeMaterialName(unsigned logid);

    VPSMap::const_iterator getVolumePhysicsScorer(size_t logid)
    {
      auto it = m_logVolID2physcorer.find(logid);
      assert(it!=m_logVolID2physcorer.end());
      return it;
    }

    std::shared_ptr<PrimaryGun> m_gun;


  private:
    friend class Singleton<GeoManager>;

    GeoManager();
    ~GeoManager();

    // the name is unique
    std::map<std::string /*material name*/, std::shared_ptr<MaterialPhysics> > m_globelPhysics;
    std::map<std::string /*scorer name*/, std::shared_ptr<Scorer> >  m_globelScorers;

    //the place to manage the life time of MaterialPhysics scorers
    std::unordered_map<size_t, std::shared_ptr<VolumePhysicsScorer>> m_logVolID2physcorer;
    std::unordered_map<size_t, std::string> m_logVolID2Mateiral;
  };
}

#endif
