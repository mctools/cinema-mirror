#include "PTScororPSD.hh"

Prompt::ScororPSD::ScororPSD(const std::string &name, double xmin, double xmax, unsigned nxbins, double ymin, double ymax, unsigned nybins)
:Scoror2D("ScororPSD_"+name, Scoror::SURFACE, std::make_unique<Hist2D>(xmin, xmax, nxbins, ymin, ymax, nybins))
{}

Prompt::ScororPSD::~ScororPSD() {}

void Prompt::ScororPSD::scoreLocal(const Vector &vec, double w)
{
  m_hist->fill(vec.x(), vec.y(), w);
}

void Prompt::ScororPSD::score(Prompt::Particle &particle)
{
  PROMPT_THROW2(BadInput, m_name << " does not support score()");
}

void Prompt::ScororPSD::score(Prompt::Particle &particle, const DeltaParticle &dltpar)
{
  PROMPT_THROW2(BadInput, m_name << " does not support score()");
}
