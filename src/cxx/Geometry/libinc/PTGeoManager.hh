#ifndef Prompt_GeoManager_hh
#define Prompt_GeoManager_hh

#include <string>
#include <map>
#include "PromptCore.hh"
#include "PTMaterialPhysics.hh"
#include "PTSingleton.hh"
#include "PTScoror.hh"

namespace Prompt {

  struct VolumePhysicsScoror { // to attached to a volume
    std::shared_ptr<MaterialPhysics> physics;
    std::vector< std::shared_ptr<Scoror> >  scorors; /*scoror name, scoror*/
  };

  class GeoManager  {
  public:
    void loadFile(const std::string &loadFile);
    std::shared_ptr<MaterialPhysics> getMaterialPhysics(const std::string &name);
    std::shared_ptr<Scoror> getScoror(const std::string &name);


  private:
    friend class Singleton<GeoManager>;

    GeoManager();
    ~GeoManager();

    // the name is unique
    std::map<std::string /*material name*/, std::shared_ptr<MaterialPhysics> > m_globelPhysics;
    std::map<std::string /*scoror name*/, std::shared_ptr<Scoror> >  m_globelScorors;

    //the place to manage the life time of MaterialPhysics scorors
    std::vector<std::shared_ptr<VolumePhysicsScoror>> m_volphyscoror;
  };
}

#endif
