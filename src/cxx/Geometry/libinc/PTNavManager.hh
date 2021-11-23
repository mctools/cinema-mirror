#ifndef Prompt_NavManager_hh
#define Prompt_NavManager_hh

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
    bool hasPropagateScoror() {return m_matphysscor->propagate_scorors.size(); };
    void scoreExit(Particle &particle);
    void scoreSurface(const Vector &particle, double w);

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
