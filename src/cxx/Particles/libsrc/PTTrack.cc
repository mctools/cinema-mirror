#include "PTTrack.hh"
#include "PTParticle.hh"

Prompt::Track::Track(Prompt::Particle &&particle, size_t eventid, size_t motherid, bool saveSpaceTime)
:Particle(std::move(particle)), m_eventid(eventid), m_motherid(motherid), m_totLength(0.),
m_saveSpaceTime(saveSpaceTime)
{
  update();
}

Prompt::Track::~Track() {}
