#include <limits>

#include "PTNCrystalScat.hh"
#include "PTUnitSystem.hh"
#include "PTRandCanonical.hh"

bool Prompt::NCrystalScat::m_ncrystal_initialised = false;

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


Prompt::NCrystalScat::NCrystalScat(const std::string &cfgstring, double bias)
:Prompt::PhysicsModel(cfgstring, const_neutron_pgd,
                      std::numeric_limits<double>::min(), 10*Prompt::Unit::eV, bias),
                      m_scat(NCrystal::createScatter(cfgstring))
{
  m_oriented = m_scat.isOriented();

  if(!m_ncrystal_initialised)
  {
    //This checks that the included NCrystal headers and the linked NCrystal
    //library are from the same release of NCrystal:
    NCrystal::libClashDetect();

    //set the generator for ncrystal
    NCrystal::setDefaultRNG(NCrystal::makeSO<SingletonPTRandWrapper>());
  }
  m_ncrystal_initialised = true;
  if(Unit::eV != 1.)
    PROMPT_THROW(CalcError, "The default unit of NCrystal is eV");

}

Prompt::NCrystalScat::~NCrystalScat()
{
  std::cout<<"Destructing scattering physics " << m_modelName <<std::endl;
}


double Prompt::NCrystalScat::getCrossSection(double ekin) const
{
  if( m_scat.isOriented() ) {
    PROMPT_THROW(CalcError, "no incident direction, material can not be oriented");
  }
  else
  {
    auto xsect = m_scat.crossSectionIsotropic( NCrystal::NeutronEnergy(ekin) );
    return xsect.get()*m_bias*Unit::barn;
  }
}

double Prompt::NCrystalScat::getCrossSection(double ekin, const Prompt::Vector &dir) const
{
  NCrystal::CrossSect xsect;
  if( m_scat.isOriented() ) {
    xsect = m_scat.crossSection( NCrystal::NeutronEnergy(ekin), {dir.x(), dir.y(), dir.z()} );
  }
  else
  {
    xsect = m_scat.crossSectionIsotropic( NCrystal::NeutronEnergy(ekin) );
  }
  return xsect.get()*m_bias*Unit::barn;
}


void Prompt::NCrystalScat::generate(double ekin, const Prompt::Vector &dir, double &final_ekin, Prompt::Vector &final_dir) const
{
  auto outcome1 = m_scat.sampleScatter( NCrystal::NeutronEnergy(ekin), {dir.x(), dir.y(), dir.z()});
  final_ekin = outcome1.ekin.get();
  auto &outdir = outcome1.direction;
  final_dir.x() = outdir[0];
  final_dir.y() = outdir[1];
  final_dir.z() = outdir[2];
}
