#ifndef Prompt_ModeratorGun_hh
#define Prompt_ModeratorGun_hh

#include "PromptCore.hh"
#include "PTPrimaryGun.hh"
#include "PTVector.hh"
#include "PTRandCanonical.hh"

namespace Prompt {
  class ModeratorGun : public PrimaryGun {
  public:
    //source size consist of 6 numbers x_front, y_front, z_front, x_back, y_back, z_back
    ModeratorGun(const Particle &aParticle, std::array<double, 6> sourceSize);
    virtual ~ModeratorGun();
    virtual void samplePosDir(Vector &pos, Vector &dir) override;

  protected:
    std::array<double, 6> m_sourceSize;
  };
}


#endif
