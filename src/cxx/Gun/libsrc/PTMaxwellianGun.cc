#include "PTMaxwellianGun.hh"

Prompt::MaxwellianGun::MaxwellianGun(const Particle &aParticle, double temperature, std::array<double, 6> sourceSize)
:ModeratorGun(aParticle, sourceSize), m_kT(temperature*const_boltzmann)
{ }

Prompt::MaxwellianGun::~MaxwellianGun()
{ }

void Prompt::MaxwellianGun::sampleEnergy(double &ekin)
{
  double cosr = cos(M_PI*0.5*m_rng.generate());
  ekin = m_kT*(-log(m_rng.generate())-log(m_rng.generate())*cosr*cosr);
}
