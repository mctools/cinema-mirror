#ifndef Prompt_Launcher_hh
#define Prompt_Launcher_hh

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

#include "PromptCore.hh"
#include "PTSingleton.hh"
#include "PTPrimaryGun.hh"
#include "PTActiveVolume.hh"
#include "PTStackManager.hh"
namespace Prompt {
  class Launcher {
  public:
    void go(uint64_t numParticle, double printPrecent, bool recordTrj=false, bool timer=true, bool save2Disk=true);
    size_t goWithSecondStack(uint64_t numParticle);

    void loadGeometry(const std::string &geofile); 

    void simOneEvent(bool recordTrj);

    // void setWorld(); //for c++ debug

    void setSeed(uint64_t seed);
    uint64_t getSeed() { return Singleton<SingletonPTRand>::getInstance().getSeed(); }
    void setGun(std::shared_ptr<PrimaryGun> gun) { m_gun=gun; }
    void setGun(const char* cfg);
    std::shared_ptr<PrimaryGun> getGun() const { return m_gun; }
    const std::vector<Vector> &getTrajectory() { return m_trajectory; }
    size_t getTrajSize() { return m_trajectory.size(); }
    void copyCurrentParticle(Particle &p) const { p = *m_curParicle.get(); };


  private:
    friend class Singleton<Launcher>;
    Launcher();
    ~Launcher();
    std::shared_ptr<PrimaryGun> m_gun;
    std::unique_ptr<Particle> m_curParicle;
    std::vector<Vector> m_trajectory;
    ActiveVolume &m_activeVolume;
    StackManager &m_stackManager;

  };
}
#endif
