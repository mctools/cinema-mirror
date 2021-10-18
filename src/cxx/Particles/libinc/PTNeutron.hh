#ifndef Prompt_Neutron_hh
#define Prompt_Neutron_hh

#include "PTParticle.hh"
#include "PTMath.hh"
//! Neutron is neutron with pgd code of 2112 by defult. Proton (2212) is also supported.
//! m_erest is in the unit of eV*c^2
//! fixme: support Gamma (22) as well.
namespace Prompt {
  class Neutron : public Particle {
  public:
    Neutron();
    Neutron(double ekin, const Vector& dir, const Vector& pos);
    virtual ~Neutron(){};
  };
}

inline Prompt::Neutron::Neutron()
:Particle()
{
   m_pgd = 2112;
   m_rest_mass = const_neutron_mass_evc2;
}

inline Prompt::Neutron::Neutron(double ekin, const Vector& dir, const Vector& pos)
:Particle(ekin, dir, pos)
{
  m_pgd = 2112;
  m_rest_mass = const_neutron_mass_evc2;
}

#endif
