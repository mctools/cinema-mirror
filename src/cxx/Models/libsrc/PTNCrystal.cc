#include <limits>

#include "PTNCrystal.hh"
#include "PTUnitSystem.hh"

Prompt::PTNCrystal::PTNCrystal(const std::string &cfgstring)
:Prompt::PhysicsModel("NCrystal"), m_scat(NCrystal::createScatter(cfgstring))
{
  m_supportPGD = const_neutron_pgd;
  m_minEkin = std::numeric_limits<double>::min();
  m_maxEkin = 10*Prompt::Unit::eV;

  //This checks that the included NCrystal headers and the linked NCrystal
  //library are from the same release of NCrystal:
  NCrystal::libClashDetect();

}

Prompt::PTNCrystal::~PTNCrystal()
{

}

bool Prompt::PTNCrystal::applicable(unsigned pgd, double ekin) const
{

return 0;
}

double Prompt::PTNCrystal::getCrossSection(double ekin) const
{
  return 0;


}

void Prompt::PTNCrystal::generate(double &ekin, Prompt::Vector &dir) const
{

}

void Prompt::PTNCrystal::generate(double ekin, const Prompt::Vector &dir, double &final_ekin, Prompt::Vector &final_dir) const
{

}
