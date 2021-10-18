#ifndef Prompt_PTNCrystal_hh
#define Prompt_PTNCrystal_hh

#include <string>
#include "PromptCore.hh"
#include "PTModelManager.hh"

namespace Prompt {

  class PTNCrystal  : public PhysicsModel {
  public:
    PTNCrystal();
    ~PTNCrystal();

    virtual bool applicable(unsigned pgd, double ekin) const override;
    virtual double getCrossSection(double ekin) const override;
    virtual void generate(double &ekin, Vector &dir) const override;
    virtual void generate(double ekin, const Vector &dir, double &final_ekin, Vector &final_dir) const override;

  private:
  };

}

#endif
