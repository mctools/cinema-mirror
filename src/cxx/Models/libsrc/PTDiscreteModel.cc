#include "PTDiscreteModel.hh"

Prompt::DiscreteModel::DiscreteModel(const std::string &name, double bias)
:PhysicsModel(name), m_bias(bias)
{ }

Prompt::DiscreteModel::DiscreteModel(const std::string &name, unsigned gdp, double emin, double emax, double bias)
:PhysicsModel(name, gdp, emin, emax), m_bias(bias)
{}

Prompt::DiscreteModel::~DiscreteModel() { }
