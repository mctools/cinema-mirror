#include "PTPrimaryGun.hh"


Prompt::Particle Prompt::PrimaryGun::generate()
{
  sampleEnergy(m_ekin);
  samplePosDir(m_pos, m_dir);
  m_eventid++;
  return *this;
}
