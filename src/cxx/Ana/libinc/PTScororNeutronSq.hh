#ifndef Prompt_ScororNeutronSq_hh
#define Prompt_ScororNeutronSq_hh

#include "PromptCore.hh"
#include "PTScoror.hh"
#include "PTHist1D.hh"

namespace Prompt {

  class ScororNeutronSq  : public Scoror {
  public:
    ScororNeutronSq(const Vector &samplePos, const Vector &refDir, double sourceSampleDist, double qmin, double qmax, unsigned numbin, bool kill=true, bool linear=true);
    virtual ~ScororNeutronSq();
    virtual void score(Particle &particle) override; //make it in the base with delta track infomation
    void save(const std::string &fname);
  private:
    const Vector m_samplePos, m_refDir;
    const double m_sourceSampleDist;
    std::unique_ptr<Hist1D> m_hist;
    bool m_kill;
  };
}



#endif
