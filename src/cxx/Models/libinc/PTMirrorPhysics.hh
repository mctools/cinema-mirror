#ifndef Prompt_MirrorPhyiscs_hh
#define Prompt_MirrorPhyiscs_hh

#include <string>

#include "PromptCore.hh"
#include "PTLookUpTable.hh"
#include "PTDiscreteModel.hh"

#include "NCrystal/NCrystal.hh"

namespace Prompt {

  class MirrorPhyiscs  : public DiscreteModel {
    public:
      MirrorPhyiscs(double mvalue, double weightCut = 1e-5);
      virtual ~MirrorPhyiscs() override;

      virtual double getCrossSection(double ekin) const override;
      virtual double getCrossSection(double ekin, const Vector &dir) const override;
      virtual void generate(double ekin, const Vector &nDirInLab, double &final_ekin, Vector &reflectionNor, double &scaleWeight) const override;

    private:
      std::shared_ptr<LookUpTable> m_table;
      double m_wcut;

  };

}

#endif
