#include <limits>

#include "PTNCrystalAbs.hh"
#include "PTUnitSystem.hh"
#include "PTRandCanonical.hh"


Prompt::NCrystalAbs::NCrystalAbs(const std::string &cfgstring, double bias)
:Prompt::PhysicsModel(cfgstring, const_neutron_pgd,
                      std::numeric_limits<double>::min(), 10*Prompt::Unit::eV, bias),
                      m_abs(NCrystal::createAbsorption(cfgstring))
{
  if( m_abs.isOriented() ) {
    PROMPT_THROW(CalcError, "Absorption process is not oriented");
  }
}

Prompt::NCrystalAbs::~NCrystalAbs()
{
  std::cout<<"Destructing absorption physics " << m_modelName <<std::endl;
}


double Prompt::NCrystalAbs::getCrossSection(double ekin) const
{
    return m_abs.crossSectionIsotropic(NCrystal::NeutronEnergy(ekin)).get()*Unit::barn;
}

double Prompt::NCrystalAbs::getCrossSection(double ekin, const Prompt::Vector &dir) const
{
  NCrystal::CrossSect xsect = m_abs.crossSectionIsotropic( NCrystal::NeutronEnergy(ekin) );
  return xsect.get()*Unit::barn;
}


void Prompt::NCrystalAbs::generate(double &ekin, Prompt::Vector &dir) const
{
  PROMPT_THROW(CalcError, "not yet implemented");
}

void Prompt::NCrystalAbs::generate(double ekin, const Prompt::Vector &dir, double &final_ekin, Prompt::Vector &final_dir) const
{
  final_ekin=-1.;
}
