#ifndef Prompt_ScororVolFlux_hh
#define Prompt_ScororVolFlux_hh

#include "PromptCore.hh"
#include "PTScoror.hh"

namespace Prompt {

  class ScororVolFlux  : public Scoror1D {
  public:
    ScororVolFlux(const std::string &name, double xmin, double xmax,
                  unsigned nxbins, bool linear, double volme);
    virtual ~ScororVolFlux();
    virtual void scoreLocal(const Vector &vec, double w) override;
    virtual void score(Particle &particle) override;
    virtual void score(Particle &particle, const DeltaParticle &dltpar) override;
  private:
    double m_iVol;
  };
}
#endif
