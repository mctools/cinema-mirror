#ifndef Prompt_ScororPSD_hh
#define Prompt_ScororPSD_hh

#include "PromptCore.hh"
#include "PTScoror.hh"

namespace Prompt {

  class ScororPSD  : public Scoror2D {
  public:
    ScororPSD(const std::string &name, double xmin, double xmax, unsigned nxbins, double ymin, double ymax, unsigned nybins);
    virtual ~ScororPSD();
    virtual void scoreLocal(const Vector &vec, double w) override;
    virtual void score(Particle &particle) override;
    virtual void score(Particle &particle, const DeltaParticle &dltpar) override;
  };
}
#endif
