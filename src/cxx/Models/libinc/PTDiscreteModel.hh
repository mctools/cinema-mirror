#ifndef Prompt_DiscreteModel_hh
#define Prompt_DiscreteModel_hh

#include <string>

#include "PromptCore.hh"
#include "PTPhysicsModel.hh"
#include <memory>

#include "NCrystal/NCrystal.hh"

namespace NCrystal {
    class Scatter;
}

namespace Prompt {

  class DiscreteModel  : public PhysicsModel {
  public:
    DiscreteModel(const std::string &name, double bias=1.0);
    DiscreteModel(const std::string &name, unsigned gdp, double emin, double emax, double bias=1.0);
    virtual ~DiscreteModel();

  protected:
    double m_bias;
  };

}

#endif
