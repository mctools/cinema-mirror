#ifndef Prompt_VolumePhysicsScorer_hh
#define Prompt_VolumePhysicsScorer_hh

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
#include "PTSurfaceProcess.hh"
#include "PTScorer.hh"

namespace Prompt {
    // A VolumePhysicsScorer should attach to a unique volume by volume idx
    // So VolMap is a map for all the volumes and its VolumePhysicsScorer 
    struct VolumePhysicsScorer { 
        //bulk physics
        std::shared_ptr<BulkMaterialProcess> bulkMaterialProcess; 
        std::shared_ptr<SurfaceProcess> surfaceProcess; //surface physics
        std::vector< std::shared_ptr<Scorer> >  scorers; /*Scorer*/

        std::vector< std::shared_ptr<Scorer> >  surface_scorers;
        std::vector< std::shared_ptr<Scorer> >  entry_scorers;
        std::vector< std::shared_ptr<Scorer> >  propagate_scorers;
        std::vector< std::shared_ptr<Scorer> >  exit_scorers;
        std::vector< std::shared_ptr<Scorer> >  absorb_scorers;
        
        // sort scorers into the five types of scorer vectors
        void sortScorers();
    };
    

    using VolMap = std::unordered_map<size_t, std::shared_ptr<VolumePhysicsScorer>>;
    using CfgPhysMap = std::unordered_map<std::string /*material name*/, std::shared_ptr<BulkMaterialProcess> > ;
    using CfgScorerMap = std::unordered_map<std::string /*scorer name*/, std::shared_ptr<Scorer> > ;

    class ResourceManager {
        public:
            ResourceManager() {};
            ~ResourceManager() = default;

            std::shared_ptr<VolumePhysicsScorer> addNewVolume(size_t id);
            std::shared_ptr<VolumePhysicsScorer> getVolume(size_t id);
        
        
        private:
            VolMap m_volumes;
            CfgPhysMap m_globelPhysics;
            CfgScorerMap m_globelScorers;
    };

}
#endif
