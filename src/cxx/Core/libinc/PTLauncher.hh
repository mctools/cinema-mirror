#ifndef Prompt_Launcher_hh
#define Prompt_Launcher_hh

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

#include "PromptCore.hh"
#include "PTSingleton.hh"
#include "PTPrimaryGun.hh"

namespace Prompt {
  class Launcher {
  public:
    void go(uint64_t numParticle, double printPrecent, bool recordTrj=false, bool timer=true);
    void loadGeometry(const std::string &geofile);
    void setSeed(uint64_t seed) { Singleton<SingletonPTRand>::getInstance().setSeed(seed); }
    uint64_t getSeed() { return Singleton<SingletonPTRand>::getInstance().getSeed(); }
    void setGun(std::shared_ptr<PrimaryGun> gun) { m_gun=gun; }
    const std::vector<Vector> &getTrajectory() { return m_trajectory; }
    size_t getTrajSize() { return m_trajectory.size(); }

  private:
    friend class Singleton<Launcher>;
    Launcher();
    ~Launcher();
    std::shared_ptr<PrimaryGun> m_gun;
    std::vector<Vector> m_trajectory;
  };
}
#endif
