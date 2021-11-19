#ifndef Prompt_NCrystalScat_hh
#define Prompt_NCrystalScat_hh

#include <string>

#include "PromptCore.hh"
#include "PTPhysicsModel.hh"
#include <memory>

#include "NCrystal/NCrystal.hh"

namespace NCrystal {
    class Scatter;
}

namespace Prompt {

  //NCrystalScat is in fact a scatterer of NCrystal
  //Physics model should be initialised from material

  class NCrystalScat  : public PhysicsModel {
  public:
    NCrystalScat(const std::string &cfgstring, double bias=1.0);
    ~NCrystalScat();

    virtual double getCrossSection(double ekin) const override;
    virtual double getCrossSection(double ekin, const Vector &dir) const override;
    virtual void generate(double &ekin, Vector &dir) const override;
    virtual void generate(double ekin, const Vector &dir, double &final_ekin, Vector &final_dir) const override;

  private:
    mutable  NCrystal::Scatter m_scat;
    static bool m_ncrystal_initialised;
  };

}

#endif
