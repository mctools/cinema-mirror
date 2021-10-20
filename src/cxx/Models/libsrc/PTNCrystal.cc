#include <limits>

#include "PTNCrystal.hh"
#include "PTUnitSystem.hh"

Prompt::PTNCrystal::PTNCrystal(const std::string &cfgstring)
:Prompt::PhysicsModel("NCrystal"), m_scat(NCrystal::createScatter(cfgstring))
{
  //fixme:
  // NC::setDefaultRNG(rng);

  m_supportPGD = const_neutron_pgd;
  m_minEkin = std::numeric_limits<double>::min();
  m_maxEkin = 10*Prompt::Unit::eV;

  m_oriented = m_scat.isOriented();

  //This checks that the included NCrystal headers and the linked NCrystal
  //library are from the same release of NCrystal:
  NCrystal::libClashDetect();
  if(Unit::eV != 1.)
    PROMPT_THROW(CalcError, "The default unit of NCrystal is eV");
}

Prompt::PTNCrystal::~PTNCrystal()
{

}


double Prompt::PTNCrystal::getCrossSection(double ekin) const
{
  if( m_scat.isOriented() ) {
    PROMPT_THROW(CalcError, "material is oriented");
  }
  else
  {
    auto xsect = m_scat.crossSectionIsotropic( NCrystal::NeutronEnergy(ekin) );
    return xsect.get()*Unit::barn;
  }
}

double Prompt::PTNCrystal::getCrossSection(double ekin, const Prompt::Vector &dir) const
{
  NCrystal::CrossSect xsect;
  if( m_scat.isOriented() ) {
    xsect = m_scat.crossSection( NCrystal::NeutronEnergy(ekin), {dir.x(), dir.y(), dir.z()} );
  }
  else
  {
    xsect = m_scat.crossSectionIsotropic( NCrystal::NeutronEnergy(ekin) );
  }
  return xsect.get()*Unit::barn;
}


void Prompt::PTNCrystal::generate(double &ekin, Prompt::Vector &dir) const
{
  PROMPT_THROW(CalcError, "not yet implemented");
}

void Prompt::PTNCrystal::generate(double ekin, const Prompt::Vector &dir, double &final_ekin, Prompt::Vector &final_dir) const
{
  auto outcome1 = m_scat.sampleScatter( NCrystal::NeutronEnergy(ekin), {dir.x(), dir.y(), dir.z()});
  final_ekin = outcome1.ekin.get();
  auto &outdir = outcome1.direction;
  final_dir.x() = outdir[0];
  final_dir.y() = outdir[1];
  final_dir.z() = outdir[2];
}
