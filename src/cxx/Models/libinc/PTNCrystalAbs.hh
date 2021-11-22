#ifndef Prompt_NCrystalAbs_hh
#define Prompt_NCrystalAbs_hh

#include <string>

#include "PromptCore.hh"
#include "PTDiscreteModel.hh"
#include <memory>

#include "NCrystal/NCrystal.hh"

namespace NCrystal {
    class Scatter;
}

namespace Prompt {

  //NCrystalAbs is in fact a scatterer of NCrystal
  //Physics model should be initialised from material

  class NCrystalAbs  : public DiscreteModel {
  public:
    NCrystalAbs(const std::string &cfgstring, double bias=1.0);
    virtual ~NCrystalAbs();

    virtual double getCrossSection(double ekin) const override;
    virtual double getCrossSection(double ekin, const Vector &dir) const override;
    virtual void generate(double ekin, const Vector &dir, double &final_ekin, Vector &final_dir, double &scaleWeight) const override;

  private:
    mutable  NCrystal::Absorption m_abs;
  };

}

#endif
