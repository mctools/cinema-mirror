#ifndef Prompt_NavManager_hh
#define Prompt_NavManager_hh

#include <string>
#include <map>
#include "PromptCore.hh"
#include "PTModelCollection.hh"
#include "PTSingleton.hh"
#include "PTParticle.hh"
#include "PTMaterial.hh"


#include <VecGeom/management/GeoManager.h>

namespace Prompt {

  class NavManager  {
  public:
    bool proprogate(Particle &particle, bool verbose = true);
    void locateLogicalVolume(const Vector &p);

  private:
    friend class Singleton<NavManager>;
    NavManager();
    ~NavManager();

    vecgeom::GeoManager &m_geo;
    vecgeom::LogicalVolume *m_currVolume;
    Prompt::Material *m_matphys;
    vecgeom::NavigationState *m_currState, *m_nextState;
  };

}

#endif
