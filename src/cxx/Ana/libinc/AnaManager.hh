#ifndef Prompt_AnaManager_hh
#define Prompt_AnaManager_hh

#include <string>
#include <map>
#include "PromptCore.hh"
#include "PTSingleton.hh"
#include "Hist1D.hh"
#include "Hist2D.hh"

namespace Prompt {

  enum AnalysisType {
    FIRST_HIT_XYPOS,
    VOLUME_FLUX,
    SURFACE_FLUX,
    SURFACE_CURRENT
  };

  struct name_t {
    /* data */
  };

  class AnaManager  {
  public:
    //used in GeoManager::loadFile
    void bookScorer(std::string volumeName, AnalysisType type);
    void initScorer(size_t id, AnalysisType type);

    void scorer();

  private:
    friend class Singleton<AnaManager>;
    AnaManager();
    ~AnaManager();
    std::multimap<std::string, size_t> m_volNameID;

    std::multimap<size_t, AnalysisType> m_volType;
    std::multimap<size_t, Hist1D> m_vol1D;
    std::multimap<size_t, Hist2D> m_vol2D;
  };
}

#endif
