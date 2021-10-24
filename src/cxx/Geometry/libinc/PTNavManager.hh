#ifndef Prompt_NavManager_hh
#define Prompt_NavManager_hh

#include <string>
#include <map>
#include "PromptCore.hh"
#include "PTModelCollection.hh"
#include "PTSingleton.hh"
#include "PTParticle.hh"
#include "PTMaterial.hh"

#include "Hist2D.hh"


#include <VecGeom/management/GeoManager.h>

namespace Prompt {

  class NavManager  {
  public:
    //return false if the track is terminated, i.e. exist world
    bool proprogateInAVolume(Particle &particle, bool verbose = true);
    void locateLogicalVolume(const Vector &p);
    bool exitWorld();
    void setupVolumePhysics();
    size_t getVolumeID();

  private:
    friend class Singleton<NavManager>;
    NavManager();
    ~NavManager();

    vecgeom::GeoManager &m_geo;
    const vecgeom::LogicalVolume *m_currVolume;
    Prompt::Material *m_matphys;
    // NavigationState is NavStateIndex when VECGEOM_USE_NAVINDEX is enabled
    // It is NavStatePath otherwise
    vecgeom::NavigationState *m_currState, *m_nextState;
    Hist2D *m_hist2d;

  };

}

#endif
