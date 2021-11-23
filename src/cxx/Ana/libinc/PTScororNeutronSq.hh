#ifndef Prompt_ScororNeutronSq_hh
#define Prompt_ScororNeutronSq_hh

#include "PromptCore.hh"
#include "PTScoror.hh"

namespace Prompt {

  class ScororNeutronSq  : public Scoror1D {
  public:
    ScororNeutronSq(const std::string &name, const Vector &samplePos, const Vector &refDir, double sourceSampleDist, double qmin, double qmax, unsigned numbin, bool kill=true, bool linear=true);
    virtual ~ScororNeutronSq();
    virtual void scoreLocal(const Vector &vec, double w) override;
    virtual void score(Particle &particle) override;
    virtual void score(Particle &particle, const DeltaParticle &dltpar) override;
  private:
    const Vector m_samplePos, m_refDir;
    const double m_sourceSampleDist;
    bool m_kill;
  };
}
#endif
