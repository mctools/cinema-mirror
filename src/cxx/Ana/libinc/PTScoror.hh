#ifndef Prompt_Scoror_hh
#define Prompt_Scoror_hh

#include "PromptCore.hh"
#include "PTParticle.hh"

namespace Prompt {

  class Scoror  {
  public:
    Scoror(const std::string& name) : m_name(name) {};
    virtual ~Scoror() {std::cout<<"Destructing scoror " << m_name <<std::endl;};
    const std::string &getName() { return m_name; }
    virtual void score(Particle &particle, bool kill) = 0;

  private:
    const std::string m_name;

  };
}

#endif
