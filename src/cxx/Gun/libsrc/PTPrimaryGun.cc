#include "PTPrimaryGun.hh"


Prompt::Particle Prompt::PrimaryGun::generate()
{
  sampleEnergy(m_ekin);
  samplePosDir(m_pos, m_dir);
  m_eventid++;
  m_weight=1.;
  m_alive=true;
  return *this;
}
