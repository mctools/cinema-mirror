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
#include "PTResourceManager.hh"

namespace Prompt {

  class GeoManager  {
  public:
    void initFromGDML(const std::string &loadFile);
    std::shared_ptr<PrimaryGun> m_gun;


  private:
    friend class Singleton<GeoManager>;
    GeoManager();
    ~GeoManager();

    void setupNavigator();

    ResourceManager &m_resman;
  };
}

#endif
