#ifndef Prompt_ModelCollection_hh
#define Prompt_ModelCollection_hh

#include <string>
#include <memory>
#include "PromptCore.hh"
#include "PTVector.hh"
#include "PTRandCanonical.hh"
#include "PTDiscreteModel.hh"

namespace Prompt {

  class PhysicsModel;

  struct XSCache {
    double ekin;
    Vector dir;
    std::vector<double> cache_xs;
    std::vector<double> bias;
    double selectedBias;
    double tot;
  };

  // This class is used to represent a collection of models for a material.
  // Only discrete models for now.

  class ModelCollection  {
  public:
    ModelCollection();
    virtual ~ModelCollection();

    double totalCrossSection(double ekin, const Vector &dir) const;
    void sample(double ekin, const Vector &dir, double &final_ekin, Vector &final_dir, double &scaleWeight) const;

    void addPhysicsModel(const std::string &cfg);
    bool sameInquiryAsLastTime(double ekin, const Vector &dir) const;

    double calculateWeight(double lengthRho, bool selBiase);

  private:
    std::vector<std::shared_ptr<DiscreteModel> > m_models;
    mutable XSCache m_cache;
    bool m_oriented;

    SingletonPTRand &m_rng;

    //fixme: cache
  };
}

#endif
