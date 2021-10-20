#ifndef Prompt_Material_hh
#define Prompt_Material_hh

#include <string>
#include "PromptCore.hh"
#include "PTModelCollection.hh"

namespace Prompt {

  class Material  {
  public:
    Material();
    virtual ~Material();

    double macroCrossSection(double ekin, const Prompt::Vector &dir);
    double sampleStepLength(double ekin, const Prompt::Vector &dir);
    void addComposition(const std::string &cfg);

  private:
    double calNumDensity(const std::string &cfg);
    PTRand &m_rng;
    std::unique_ptr<ModelCollection> m_model;
    double m_numdensity;
  };

}

#endif
