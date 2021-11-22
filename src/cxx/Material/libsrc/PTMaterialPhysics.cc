#include "PTMaterialPhysics.hh"
#include <cmath>
#include <limits>
#include "NCrystal/NCrystal.hh"

Prompt::MaterialPhysics::MaterialPhysics()
:m_rng(Singleton<SingletonPTRand>::getInstance()), m_model(std::make_unique<ModelCollection>()),
m_numdensity(0.)
{
}

Prompt::MaterialPhysics::~MaterialPhysics()
{
}

double Prompt::MaterialPhysics::macroCrossSection(double ekin, const Prompt::Vector &dir)
{
  return m_numdensity*m_model->totalCrossSection(ekin, dir);
}

void Prompt::MaterialPhysics::sampleFinalState(double ekin, const Vector &dir, double &final_ekin, Vector &final_dir, double &scaleWeight)
{
  m_model->sample(ekin, dir, final_ekin, final_dir, scaleWeight);
}

double Prompt::MaterialPhysics::sampleStepLength(double ekin, const Prompt::Vector &dir)
{
  double mxs = macroCrossSection(ekin, dir);
  if(mxs)
    return -log(m_rng.generate())/mxs;
  else
    return std::numeric_limits<double>::max();
}

double Prompt::MaterialPhysics::calNumDensity(const std::string &cfg)
{
  NCrystal::MatCfg matcfg(cfg);
  auto info = NCrystal::createInfo(matcfg);
  if(info->hasNumberDensity())
    return info->getNumberDensity().get() / Unit::Aa3;
  else
  {
    PROMPT_THROW2(CalcError, "material has no material " << cfg);
    return 0.;
  }
}

void Prompt::MaterialPhysics::addComposition(const std::string &cfg)
{
  m_model->addPhysicsModel(cfg);
  m_numdensity += calNumDensity(cfg);
}
