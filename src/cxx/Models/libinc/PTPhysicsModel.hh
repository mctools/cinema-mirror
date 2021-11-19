#ifndef Prompt_PhysicsModel_hh
#define Prompt_PhysicsModel_hh

#include <string>
#include <memory>
#include "PromptCore.hh"
#include "PTVector.hh"

#include "NCrystal/NCrystal.hh"
#include "PTRandCanonical.hh"

namespace Prompt {

  class SingletonPTRandWrapper : public NCrystal::RNGStream{
  public:
    SingletonPTRandWrapper()
    :NCrystal::RNGStream(), m_ptrng(Prompt::Singleton<Prompt::SingletonPTRand>::getInstance())
    {}
    virtual ~SingletonPTRandWrapper() override {}

    double actualGenerate() override {return m_ptrng.generate(); }

    //For the sake of example, we wrongly claim that this generator is safe and
    //sensible to use multithreaded (see NCRNG.hh for how to correctly deal with
    //MT safety, RNG states, etc.):
    bool useInAllThreads() const override { return true; }
  private:
    Prompt::SingletonPTRand &m_ptrng;
  };

  class PhysicsModel {
  public:
    PhysicsModel(const std::string &name, double bias=1.)
     :m_modelName(name), m_oriented(false), m_bias(bias)  {};

    PhysicsModel(const std::string &name, unsigned gdp,
                 double emin, double emax, double bias=1.)
     :m_modelName(name), m_supportPGD(gdp), m_minEkin(emin),
      m_maxEkin(emax), m_oriented(false), m_bias(bias)  {};

    virtual ~PhysicsModel() {};

    bool applicable(unsigned pgd) const { return m_supportPGD==pgd; };
    bool isOriented() {return m_oriented;};
    void getEnergyRange(double &ekinMin, double &ekinMax) {
      m_minEkin = ekinMin;
      m_maxEkin = ekinMax;
    };

    void setEnergyRange(double ekinMin, double ekinMax) {
      ekinMin = m_minEkin;
      ekinMax = m_maxEkin;
    };

    virtual bool applicable(unsigned pgd, double ekin) const {
      return pgd==m_supportPGD && (ekin > m_minEkin && ekin < m_maxEkin);
    };

    virtual double getCrossSection(double ekin) const = 0;
    virtual double getCrossSection(double ekin, const Vector &dir) const = 0;
    virtual void generate(double &ekin, Vector &dir) const = 0;
    virtual void generate(double ekin, const Vector &dir, double &final_ekin, Vector &final_dir) const = 0;

  protected:
    std::string m_modelName;
    unsigned m_supportPGD;
    double m_minEkin, m_maxEkin;
    bool m_oriented;
    double m_bias;
  };

}

#endif
