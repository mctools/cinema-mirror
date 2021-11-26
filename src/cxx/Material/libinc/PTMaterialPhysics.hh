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
    double getScaleWeight(double step, bool selBiase);
    void sampleFinalState(double ekin, const Vector &dir, double &final_ekin, Vector &final_dir, double &scaleWeight);
    void addComposition(const std::string &cfg, double bias=1.0);

  private:
    double calNumDensity(const std::string &cfg);
    SingletonPTRand &m_rng;
    std::unique_ptr<ModelCollection> m_modelcoll;
    double m_numdensity;

  };

}

#endif
