#ifndef Prompt_AnaManager_hh
#define Prompt_AnaManager_hh

#include "PromptCore.hh"
#include "PTSingleton.hh"
#include "PTHist1D.hh"
#include "PTHist2D.hh"
#include "PTScoror.hh"
namespace Prompt {

  enum AnalysisType {
    POSXY,
    VOLUME_FLUX,
    SURFACE_FLUX,
    SURFACE_CURRENT
  };

  class AnaManager  {
  public:
    //used in GeoManager::loadFile
    std::unique_ptr<Scoror> createScoror(const std::string &cfg);

  private:
    friend class Singleton<AnaManager>;
    AnaManager();
    ~AnaManager();
  };
}

#endif
