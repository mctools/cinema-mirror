#ifndef Prompt_ModelManager_hh
#define Prompt_ModelManager_hh

#include <string>
#include <memory>
#include "PromptCore.hh"
#include "PTVector.hh"

namespace Prompt {

  class PhysicsModel {
  public:
    PhysicsModel() {};
    virtual ~PhysicsModel() {};
    bool applicable(unsigned pgd) const { return m_supportPGD==pgd; };
    void getEnergyRange(double &ekinMin, double &ekinMax) {
      m_minEkin = ekinMin;
      m_maxEkin = ekinMax;
    };
    void setEnergyRange(double ekinMin, double ekinMax) {
      ekinMin = m_minEkin;
      ekinMax = m_maxEkin;
    };

    virtual bool applicable(unsigned pgd, double ekin) const = 0;
    virtual double getCrossSection(double ekin) const = 0;
    virtual void generate(double &ekin, Vector &dir) const = 0;
    virtual void generate(double ekin, const Vector &dir, double &final_ekin, Vector &final_dir) const = 0;

  protected:
    std::string m_modelName;
    unsigned m_supportPGD;
    double m_minEkin, m_maxEkin;
  };

  class ModelManager  {
  public:
    ModelManager();
    ~ModelManager();
  private:
    std::vector<std::shared_ptr<PhysicsModel> > m_models;
  };
}

#endif
