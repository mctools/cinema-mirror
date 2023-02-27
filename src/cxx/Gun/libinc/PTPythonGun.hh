#ifndef Prompt_PythonGun_hh
#define Prompt_PythonGun_hh

#include "PTPrimaryGun.hh"

namespace Prompt {
  class PythonGun : public PrimaryGun {
  public:
    PythonGun(const Particle &aParticle)
    : Particle(aParticle) {  };
    virtual ~PythonGun() = default;
    virtual std::unique_ptr<Particle> generate();
    virtual void sampleEnergy(double &ekin) = 0;
    virtual void samplePosDir(Vector &pos, Vector &dir) = 0;
    virtual double getParticleWeight() { return 1.;}
    virtual double getTime() { return 0.;}
  };
}



#ifdef __cplusplus
extern "C" {
#endif

void pt_PythonGun_sampleEnergy(double &ekin);



#ifdef __cplusplus
}
#endif



#endif
