#ifndef Prompt_MaxwellianGun_hh
#define Prompt_MaxwellianGun_hh

#include "PromptCore.hh"
#include "PTModeratorGun.hh"
#include "PTVector.hh"
#include "PTRandCanonical.hh"

namespace Prompt {
  class MaxwellianGun : public ModeratorGun {
  public:
    //source size consist of 6 numbers x_front, y_front, z_front, x_back, y_back, z_back
    MaxwellianGun(const Particle &aParticle, double temperature, std::array<double, 6> sourceSize);
    virtual ~MaxwellianGun();
    virtual void sampleEnergy(double &ekin) override;

  private:
    double m_kT;


  };
}


#endif
