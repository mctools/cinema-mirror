#include "PTMaxwellianGun.hh"

Prompt::MaxwellianGun::MaxwellianGun(const Particle &aParticle, double temperature, std::array<double, 6> sourceSize)
:PrimaryGun(aParticle), m_kT(temperature*const_boltzmann), m_sourceSize(sourceSize)
{ }

Prompt::MaxwellianGun::~MaxwellianGun()
{ }

void Prompt::MaxwellianGun::sampleEnergy(double &ekin)
{
  double cosr = cos(M_PI*0.5*m_rng.generate());
  ekin = m_kT*(-log(m_rng.generate())-log(m_rng.generate())*cosr*cosr);
}

void Prompt::MaxwellianGun::samplePosDir(Vector &pos, Vector &dir)
{
  double moderator_x = (m_rng.generate()-0.5)*m_sourceSize[0];
  double moderator_y = (m_rng.generate()-0.5)*m_sourceSize[1];
  double flightPath = m_sourceSize[5]-m_sourceSize[2];
  pos = Vector{moderator_x, moderator_y, m_sourceSize[2]};

  double slit_x = (m_rng.generate()-0.5)*m_sourceSize[3];
  double slit_y = (m_rng.generate()-0.5)*m_sourceSize[4];
  dir = Vector{-moderator_x+slit_x, -moderator_y+slit_y, flightPath};
  dir = dir.unit();
}
