#ifndef Prompt_PTNCrystal_hh
#define Prompt_PTNCrystal_hh

#include <string>

#include "PromptCore.hh"
#include "PTModelManager.hh"
#include <memory>

#include "NCrystal/NCrystal.hh"

namespace NCrystal {
    class Scatter;
}

namespace Prompt {

  //PTNCrystal is in fact a scatterer of NCrystal
  //Physics model should be initialised from material

  class PTNCrystal  : public PhysicsModel {
  public:
    PTNCrystal(const std::string &cfgstring);
    ~PTNCrystal();

    virtual double getCrossSection(double ekin) const override;
    virtual double getCrossSection(double ekin, const Vector &dir) const override;
    virtual void generate(double &ekin, Vector &dir) const override;
    virtual void generate(double ekin, const Vector &dir, double &final_ekin, Vector &final_dir) const override;

  private:
    mutable  NCrystal::Scatter m_scat;
    // NCrystal::ProcImpl::ProcPtr == NCrystal::ProcImpl::Process*
  };

}

#endif
