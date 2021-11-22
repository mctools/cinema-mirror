#ifndef Prompt_MaterialPhysics_hh
#define Prompt_MaterialPhysics_hh

#include <string>
#include "PromptCore.hh"
#include "PTModelCollection.hh"

namespace Prompt {
  class MaterialPhysics  {
  public:
    MaterialPhysics();
    virtual ~MaterialPhysics();

    double macroCrossSection(double ekin, const Prompt::Vector &dir);
    double sampleStepLength(double ekin, const Prompt::Vector &dir);
    void sampleFinalState(double ekin, const Vector &dir, double &final_ekin, Vector &final_dir, double &scaleWeight);
    void addComposition(const std::string &cfg);

  private:
    double calNumDensity(const std::string &cfg);
    SingletonPTRand &m_rng;
    std::unique_ptr<ModelCollection> m_model;
    double m_numdensity;
  };

}

#endif
