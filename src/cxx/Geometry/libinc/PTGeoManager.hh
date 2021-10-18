#ifndef Prompt_GeoManager_hh
#define Prompt_GeoManager_hh

#include <string>
#include "PromptCore.hh"

namespace Prompt {

  class GeoManager  {
  public:
    GeoManager(const std::string &gdml_file);
    ~GeoManager();
  private:
  };

}

#endif
