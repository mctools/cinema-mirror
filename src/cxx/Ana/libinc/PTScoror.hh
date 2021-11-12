#ifndef Prompt_Scoror_hh
#define Prompt_Scoror_hh

#include "PromptCore.hh"
#include "PTParticle.hh"

namespace Prompt {

  enum ScororType { ENTRY, PROPAGATE, EXIT};

  class Scoror  {
  public:
    Scoror(const std::string& name, ScororType type) : m_name(name), m_type(type) {};
    virtual ~Scoror() {std::cout<<"Destructing scoror " << m_name <<std::endl;};
    const std::string &getName() { return m_name; }
    ScororType getType() { return m_type; }
    virtual void score(Particle &particle) = 0;
    virtual void score(Particle &particle, const DeltaParticle &dltpar) = 0;

  private:
    const std::string m_name;
    ScororType m_type;

  };
}

#endif
