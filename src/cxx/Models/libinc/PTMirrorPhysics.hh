#ifndef Prompt_MirrorPhyiscs_hh
#define Prompt_MirrorPhyiscs_hh

#include <string>

#include "PromptCore.hh"
#include "PTDiscreteModel.hh"
#include <memory>

#include "NCrystal/NCrystal.hh"

namespace Prompt {

  class MirrorPhyiscs  : public DiscreteModel {
  public:
    MirrorPhyiscs(const std::string &cfgstring);
    ~MirrorPhyiscs();

    virtual double getCrossSection(double ekin) const override;
    virtual double getCrossSection(double ekin, const Vector &dir) const override;
    virtual void generate(double ekin, const Vector &nDirInLab, double &final_ekin, Vector &reflectionNor, double &scaleWeight) const override;

  };

}

#endif
