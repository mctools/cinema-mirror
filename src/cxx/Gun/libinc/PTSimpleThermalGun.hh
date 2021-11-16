#ifndef Prompt_SimpleThermalGun_hh
#define Prompt_SimpleThermalGun_hh

#include "PromptCore.hh"
#include "PTParticle.hh"
#include "PTVector.hh"

namespace Prompt {
  class SimpleThermalGun : public PrimaryGun {
  public:
    SimpleThermalGun(const Particle &aParticle)
    : PrimaryGun(aParticle) {};
    virtual ~SimpleThermalGun() {};
    virtual void sampleEnergy(double &ekin) {ekin = 0.0253; };
    virtual void samplePosDir(Vector &pos, Vector &dir) { pos = Vector{0.,0.,-2000.}; dir=Vector{0.,0.,1.}; }
  };
}


#endif
