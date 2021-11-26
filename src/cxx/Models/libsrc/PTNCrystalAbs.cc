#include <limits>

#include "PTNCrystalAbs.hh"
#include "PTUnitSystem.hh"
#include "PTRandCanonical.hh"


Prompt::NCrystalAbs::NCrystalAbs(const std::string &cfgstring, double bias)
:Prompt::DiscreteModel(cfgstring, const_neutron_pgd,
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
    return m_abs.crossSectionIsotropic(NCrystal::NeutronEnergy(ekin)).get()*Unit::barn*m_bias;
}

double Prompt::NCrystalAbs::getCrossSection(double ekin, const Prompt::Vector &dir) const
{
  NCrystal::CrossSect xsect = m_abs.crossSectionIsotropic( NCrystal::NeutronEnergy(ekin) );
  return xsect.get()*Unit::barn*m_bias;
}


void Prompt::NCrystalAbs::generate(double ekin, const Prompt::Vector &dir, double &final_ekin, Prompt::Vector &final_dir, double &scaleWeight) const
{
  scaleWeight = 1.;
  final_ekin=-1.;
}
