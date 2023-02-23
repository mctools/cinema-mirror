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
#include "PTBulkMaterialProcess.hh"
#include "PTSingleton.hh"
#include "PTScorer.hh"
#include "PTPrimaryGun.hh"
#include "PTSurfaceProcess.hh"
#include "PTVolumePhysicsScorer.hh"

namespace Prompt {

  class GeoManager  {
  public:
    void initFromGDML(const std::string &loadFile);
    void steupFakePhyisc(); //for c++ debug

    std::shared_ptr<BulkMaterialProcess> getBulkMaterialProcess(const std::string &name);
    std::shared_ptr<Scorer> getScorer(const std::string &name);

    size_t numBulkMaterialProcess() {return m_globelPhysics.size();}
    size_t numScorer() {return m_globelScorers.size();}

    std::string getLogicalVolumeScorerName(unsigned logid);
    // const std::string &getLogicalVolumeMaterialName(unsigned logid);
    void writeScorer2Disk();

    VolMap::const_iterator getVolumePhysicsScorer(size_t logid)
    {
      auto it = m_logVolID2physcorer.find(logid);
      if(it==m_logVolID2physcorer.end())
        PROMPT_THROW2(CalcError, "The physics and scorer for volme " << logid
         << " is not set");
      return it;
    }

    std::shared_ptr<PrimaryGun> m_gun;


  private:
    friend class Singleton<GeoManager>;

    GeoManager();
    ~GeoManager();

    void setupNavigator();

    // the name is unique
    std::map<std::string /*material name*/, std::shared_ptr<BulkMaterialProcess> > m_globelPhysics;
    std::map<std::string /*scorer name*/, std::shared_ptr<Scorer> >  m_globelScorers;

    //the place to manage the life time of BulkMaterialProcess scorers
    VolMap m_logVolID2physcorer;
  };
}

#endif
