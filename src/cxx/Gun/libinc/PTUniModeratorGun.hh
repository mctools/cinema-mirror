#ifndef Prompt_UniModeratorGun_hh
#define Prompt_UniModeratorGun_hh

#include "PromptCore.hh"
#include "PTModeratorGun.hh"
#include "PTVector.hh"
#include "PTRandCanonical.hh"

namespace Prompt {
  class UniModeratorGun : public ModeratorGun {
  public:
    UniModeratorGun(const Particle &aParticle, double wl0, double dlt_wl, std::array<double, 6> sourceSize)
    :ModeratorGun(aParticle, sourceSize), m_wl0(wl0), m_dlt_wl(dlt_wl) {}

    virtual ~UniModeratorGun() {};
    virtual void sampleEnergy(double &ekin) override
    {
      ekin = neutronEKin2wl(m_wl0+ m_dlt_wl*(m_rng.generate()-0.5));
    }

  private:
    double m_wl0, m_dlt_wl;


  };
}


#endif
