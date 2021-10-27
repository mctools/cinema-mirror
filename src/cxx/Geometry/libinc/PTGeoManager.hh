#ifndef Prompt_GeoManager_hh
#define Prompt_GeoManager_hh

#include <string>
#include <map>
#include "PromptCore.hh"
#include "PTMaterialPhysics.hh"
#include "PTSingleton.hh"

// namespace vecgeom
// {
//   class LogicalVolume;
// }

namespace Prompt {

  class GeoManager  {
  public:
    void loadFile(const std::string &loadFile);

  private:
    friend class Singleton<GeoManager>;

    GeoManager();
    ~GeoManager();

    std::map<std::string, std::unique_ptr<MaterialPhysics> > m_volmodelmap; // the place to manage the life time of ModelCollection
  };

}

#endif
