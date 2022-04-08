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
#include "PTScoror.hh"
#include "PTPrimaryGun.hh"
#include "PTMirrorPhysics.hh"

namespace Prompt {

  struct VolumePhysicsScoror { // to attached to a volume
    std::shared_ptr<MaterialPhysics> physics; //bulk physics
    std::shared_ptr<MirrorPhyiscs> mirrorPhysics; //boundary physics
    std::vector< std::shared_ptr<Scoror> >  scorors; /*scoror name, scoror*/

    std::vector< std::shared_ptr<Scoror> >  surface_scorors;
    std::vector< std::shared_ptr<Scoror> >  entry_scorors;
    std::vector< std::shared_ptr<Scoror> >  propagate_scorors;
    std::vector< std::shared_ptr<Scoror> >  exit_scorors;
    std::vector< std::shared_ptr<Scoror> >  absorb_scorors;


    void sortScorors()
    {
      entry_scorors.clear();
      propagate_scorors.clear();
      exit_scorors.clear();

      for(auto &v : scorors)
      {
        auto type = v->getType();
        if(type==Scoror::ENTRY)
        {
          entry_scorors.push_back(v);
        }
        else if(type==Scoror::PROPAGATE)
        {
          propagate_scorors.push_back(v);
        }
        else if(type==Scoror::EXIT)
        {
          exit_scorors.push_back(v);
        }
        else if(type==Scoror::SURFACE)
        {
          surface_scorors.push_back(v);
        }
        else if(type==Scoror::ABSORB)
        {
          absorb_scorors.push_back(v);
        }
        else
          PROMPT_THROW2(BadInput, "unknown scoror type " << type);
      }
    }
  };
  using VPSMap = std::unordered_map<size_t, std::shared_ptr<VolumePhysicsScoror>>;

  class GeoManager  {
  public:
    void loadFile(const std::string &loadFile);
    std::shared_ptr<MaterialPhysics> getMaterialPhysics(const std::string &name);
    std::shared_ptr<Scoror> getScoror(const std::string &name);
    size_t numMaterialPhysics() {return m_globelPhysics.size();}
    size_t numScoror() {return m_globelScorors.size();}

    VPSMap::const_iterator getVolumePhysicsScoror(size_t id)
    {
      auto it = m_volphyscoror.find(id);
      assert(it!=m_volphyscoror.end());
      return it;
    }

    std::shared_ptr<PrimaryGun> m_gun;


  private:
    friend class Singleton<GeoManager>;

    GeoManager();
    ~GeoManager();

    // the name is unique
    std::map<std::string /*material name*/, std::shared_ptr<MaterialPhysics> > m_globelPhysics;
    std::map<std::string /*scoror name*/, std::shared_ptr<Scoror> >  m_globelScorors;

    //the place to manage the life time of MaterialPhysics scorors
    std::unordered_map<size_t, std::shared_ptr<VolumePhysicsScoror>> m_volphyscoror;
  };
}

#endif
