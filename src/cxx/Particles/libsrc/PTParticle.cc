#include "PTParticle.hh"

void Prompt::Particle::kill()
{
  m_alive = false;
}

bool Prompt::Particle::isAlive()
{
  return m_alive;
}
