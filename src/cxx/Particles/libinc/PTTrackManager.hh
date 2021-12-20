#ifndef Prompt_SingletonTrackManager_hh
#define Prompt_SingletonTrackManager_hh

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

#include "PTSingleton.hh"
#include "PTTrack.hh"
#include "PromptCore.hh"

namespace Prompt {

  class TrackManager {
  public:
    void addTrack(std::unique_ptr<Prompt::Particle> &aparticle);

  private:
    friend class Singleton<TrackManager>;
    TrackManager();
    ~TrackManager();
    std::vector<std::unique_ptr<Particle> > m_tracks;
  };
}

#endif
