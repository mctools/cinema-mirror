#ifndef Prompt_NavManager_hh
#define Prompt_NavManager_hh

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
#include "PromptCore.hh"
#include "PTModelCollection.hh"
#include "PTSingleton.hh"
#include "PTParticle.hh"
#include "PTMaterialPhysics.hh"
#include "PTScoror.hh"
#include "PTHist2D.hh"
#include <VecGeom/management/GeoManager.h>
#include "PTGeoManager.hh"

namespace Prompt {

  class NavManager  {
  public:
    //return false if the track is terminated, i.e. exist world
    bool proprogateInAVolume(Particle &particle, bool verbose = true);
    void locateLogicalVolume(const Vector &p);
    bool exitWorld();
    void setupVolumePhysics();
    size_t getVolumeID();
    std::string getVolumeName();
    const vecgeom::VPlacedVolume *getVolume();

    void scoreEntry(Particle &particle);
    void scorePropagate(Particle &particle, const DeltaParticle &dltpar);
    void scoreExit(Particle &particle);
    void scoreSurface(const Vector &particle, double w);
    void scoreAbsorb(Particle &particle);

    bool hasPropagateScoror() {return m_matphysscor->propagate_scorors.size(); };

    bool hasMirrorPhyiscs();
    bool surfacePhysics(Particle &particle);

  private:
    friend class Singleton<NavManager>;
    NavManager();
    ~NavManager();

    vecgeom::GeoManager &m_geo;
    const vecgeom::VPlacedVolume *m_currPV;
    std::shared_ptr<VolumePhysicsScoror> m_matphysscor;
    // NavigationState is NavStateIndex when VECGEOM_USE_NAVINDEX is enabled
    // It is NavStatePath otherwise
    vecgeom::NavigationState *m_currState, *m_nextState;
  };

}

#endif
