#ifndef Prompt_Track_hh
#define Prompt_Track_hh

#include <vector>
#include "PTVector.hh"
#include "PTParticle.hh"

namespace Prompt {
  class Particle;

  struct SpaceTime {
    Vector pos;
    double time; // negative time means deleted particle
    //region ???
  };

  class Track : public Particle {
  public:
    Track(Particle &&particle, size_t eventid, size_t motherid, bool saveSpaceTime=true);
    virtual  ~Track();
    virtual void moveForward(double length) override;

  private:
    void update(double length=0);
    size_t m_eventid, m_motherid; //the first particle's m_motherid is zero
    double m_totLength;
    bool m_saveSpaceTime;
    std::vector<SpaceTime> m_spacetime;
  };
}

inline void Prompt::Track::moveForward(double length)
{
  Particle::moveForward(length);
  update(length);
}

inline void Prompt::Track::update(double length)
{
  m_totLength += length;
  if(m_saveSpaceTime)
    m_spacetime.emplace_back(SpaceTime{m_pos, m_time} );
}

#endif
