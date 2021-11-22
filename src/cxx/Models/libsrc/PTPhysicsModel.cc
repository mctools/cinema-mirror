#include "PTPhysicsModel.hh"

Prompt::PhysicsModel::PhysicsModel(const std::string &name, double bias)
 :m_modelName(name), m_oriented(false), m_bias(bias)  {};

Prompt::PhysicsModel::PhysicsModel(const std::string &name, unsigned gdp,
             double emin, double emax, double bias)
 :m_modelName(name), m_supportPGD(gdp), m_minEkin(emin),
  m_maxEkin(emax), m_oriented(false), m_bias(bias)  {};

bool Prompt::PhysicsModel::applicable(unsigned pgd) const
{ return m_supportPGD==pgd; }

bool Prompt::PhysicsModel::isOriented()
{return m_oriented;}

void Prompt::PhysicsModel::getEnergyRange(double &ekinMin, double &ekinMax)
{
  m_minEkin = ekinMin;
  m_maxEkin = ekinMax;
};

void Prompt::PhysicsModel::setEnergyRange(double ekinMin, double ekinMax)
{
  ekinMin = m_minEkin;
  ekinMax = m_maxEkin;
};

bool Prompt::PhysicsModel::applicable(unsigned pgd, double ekin) const
{
  return pgd==m_supportPGD && (ekin > m_minEkin && ekin < m_maxEkin);
};
