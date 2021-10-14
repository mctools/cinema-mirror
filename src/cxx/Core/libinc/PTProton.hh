#ifndef Prompt_Proton_hh
#define Prompt_Proton_hh

#include "PTParticle.hh"
#include "PTMath.hh"

namespace Prompt {
  class Proton : public Particle {
  public:
    Proton();
    Proton(double ekin, const Vector& dir, const Vector& pos);
    virtual ~Proton(){};
  };
}

inline Prompt::Proton::Proton()
:Particle()
{
   m_pgd = 2212;
   m_rest_mass = const_proton_rest_mass;
}


inline Prompt::Proton::Proton(double ekin, const Vector& dir, const Vector& pos)
:Particle(ekin, dir, pos)
{
  m_pgd = 2212;
  m_rest_mass = const_proton_rest_mass;
}

#endif
