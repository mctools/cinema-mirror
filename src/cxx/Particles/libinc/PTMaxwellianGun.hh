#ifndef Prompt_MaxwellianGun_hh
#define Prompt_MaxwellianGun_hh

#include "PromptCore.hh"
#include "PTPrimaryGun.hh"
#include "PTVector.hh"
#include "PTRandCanonical.hh"

namespace Prompt {
  class MaxwellianGun : public PrimaryGun {
  public:
    //source size consist of 4 numbers x_front, y_front, z_front, x_back, y_back, z_back
    MaxwellianGun(const Particle &aParticle, double temperature, std::array<double, 6> sourceSize);
    virtual ~MaxwellianGun();
    virtual void sampleEnergy(double &ekin) override;
    virtual void samplePosDir(Vector &pos, Vector &dir) override;

  private:
    double m_kT;
    std::array<double, 6> m_sourceSize;


  };
}


#endif
