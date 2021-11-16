#ifndef Prompt_PrimaryGun_hh
#define Prompt_PrimaryGun_hh

#include "PromptCore.hh"
#include "PTParticle.hh"
#include "PTVector.hh"
#include "PTRandCanonical.hh"

namespace Prompt {
  class PrimaryGun : public Particle {
  public:
    PrimaryGun(const Particle &aParticle)
    : Particle(aParticle), m_rng(Singleton<SingletonPTRand>::getInstance()) {};
    virtual ~PrimaryGun() {};
    virtual Particle generate();
    virtual void sampleEnergy(double &ekin) = 0;
    virtual void samplePosDir(Vector &pos, Vector &dir) = 0;

  protected:
    SingletonPTRand &m_rng;

  };
}


#endif
