#ifndef Prompt_Scoror_hh
#define Prompt_Scoror_hh

#include "PromptCore.hh"
#include "PTParticle.hh"
#include "PTHist1D.hh"
#include "PTHist2D.hh"

namespace Prompt {


  class Scoror {
  public:
    enum ScororType {SURFACE, ENTRY, PROPAGATE, EXIT};
  public:
    Scoror(const std::string& name, ScororType type) : m_name(name), m_type(type) {};
    virtual ~Scoror() {std::cout<<"Destructing scoror " << m_name <<std::endl;};
    const std::string &getName() { return m_name; }
    ScororType getType() { return m_type; }
    virtual void scoreLocal(const Vector &vec, double w) = 0;
    virtual void score(Particle &particle) = 0;
    virtual void score(Particle &particle, const DeltaParticle &dltpar) = 0;
    virtual void save(const std::string &fname) = 0;
  protected:
    const std::string m_name;
    ScororType m_type;
  };

  class Scoror1D : public Scoror {
  public:
    Scoror1D(const std::string& name, ScororType type, std::unique_ptr<Hist1D> hist)
    : Scoror(name, type), m_hist(std::move(hist)) {};
    virtual ~Scoror1D() { save(m_name); }
    void save(const std::string &fname) { m_hist->save(fname); }
  protected:
    std::unique_ptr<Hist1D> m_hist;
  };

  class Scoror2D : public Scoror {
  public:
    Scoror2D(const std::string& name, ScororType type, std::unique_ptr<Hist2D> hist)
    : Scoror(name, type), m_hist(std::move(hist)) {};
    virtual ~Scoror2D() { save(m_name); }
    void save(const std::string &fname) { m_hist->save(fname); }
  protected:
    std::unique_ptr<Hist2D> m_hist;
  };
}

#endif
