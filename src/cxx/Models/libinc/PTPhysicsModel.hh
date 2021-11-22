#ifndef Prompt_PhysicsModel_hh
#define Prompt_PhysicsModel_hh

#include <string>
#include <memory>
#include "PromptCore.hh"
#include "PTVector.hh"

#include "NCrystal/NCrystal.hh"
#include "PTRandCanonical.hh"

namespace Prompt {

  class PhysicsModel {
  public:
    PhysicsModel(const std::string &name, double bias=1.);
    PhysicsModel(const std::string &name, unsigned gdp,
                 double emin, double emax, double bias=1.);

    virtual ~PhysicsModel() {};

    bool applicable(unsigned pgd) const;
    bool isOriented();
    void getEnergyRange(double &ekinMin, double &ekinMax) ;
    void setEnergyRange(double ekinMin, double ekinMax);
    virtual bool applicable(unsigned pgd, double ekin) const;
    virtual double getCrossSection(double ekin) const = 0;
    virtual double getCrossSection(double ekin, const Vector &dir) const = 0;
    virtual void generate(double &ekin, Vector &dir) const = 0;
    virtual void generate(double ekin, const Vector &dir, double &final_ekin, Vector &final_dir) const = 0;

  protected:
    std::string m_modelName;
    unsigned m_supportPGD;
    double m_minEkin, m_maxEkin;
    bool m_oriented;
    double m_bias;
  };

}

#endif
