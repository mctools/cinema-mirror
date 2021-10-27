#ifndef Prompt_GeoManager_hh
#define Prompt_GeoManager_hh

#include <string>
#include <map>
#include "PromptCore.hh"
#include "PTMaterialPhysics.hh"
#include "PTSingleton.hh"
#include "PTScoror.hh"

namespace Prompt {

  struct VolumePhysicsScoror {
    std::unique_ptr<MaterialPhysics>  physics;
    std::vector<std::unique_ptr<Scoror> > scorors;
  };

  class GeoManager  {
  public:
    void loadFile(const std::string &loadFile);

  private:
    friend class Singleton<GeoManager>;

    GeoManager();
    ~GeoManager();

    //the place to manage the life time of MaterialPhysics scorors
    std::map<std::string, std::unique_ptr<VolumePhysicsScoror> > m_volphyscoror;
  };
}

#endif
