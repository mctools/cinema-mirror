#ifndef Prompt_SimpleThermalGun_hh
#define Prompt_SimpleThermalGun_hh

#include "PromptCore.hh"
#include "PTParticle.hh"
#include "PTVector.hh"

namespace Prompt {
  class SimpleThermalGun : public PrimaryGun {
  public:
    SimpleThermalGun(const Particle &aParticle, double ekin=0.0253, const Vector &pos=Vector{0.,0.,-12000.}, const Vector &dir=Vector{0.,0.,1.} )
    : PrimaryGun(aParticle), m_ekin(ekin), m_pos(pos), m_dir(dir) {};
    virtual ~SimpleThermalGun() {};
    virtual void sampleEnergy(double &ekin) { ekin = m_ekin; };
    virtual void samplePosDir(Vector &pos, Vector &dir) { pos = m_pos; dir=m_dir; }
  private:
    double m_ekin;
    Vector m_pos, m_dir;
  };
}


#endif
