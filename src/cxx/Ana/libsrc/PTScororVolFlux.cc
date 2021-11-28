#include "PTScororVolFlux.hh"

Prompt::ScororVolFlux::ScororVolFlux(const std::string &name, double xmin, double xmax, unsigned nxbins, bool linear, double volme)
:Scoror1D("ScororVolFlux_"+ name, Scoror::PROPAGATE, std::make_unique<Hist1D>(xmin, xmax, nxbins, linear)), m_iVol(1./volme)
{ }

Prompt::ScororVolFlux::~ScororVolFlux() {}


void Prompt::ScororVolFlux::scoreLocal(const Vector &vec, double w)
{
  PROMPT_THROW2(BadInput, m_name << " does not support scoreLocal()");
}

void Prompt::ScororVolFlux::score(Particle &particle)
{
  PROMPT_THROW2(BadInput, m_name << " does not support score(Particle &particle)");
}

void Prompt::ScororVolFlux::score(Particle &particle, const DeltaParticle &dltpar)
{
  //w=m_iVol*dltpar.dlt_pos.mag()*particle.getWeight()
  m_hist->fill(particle.getEKin()-dltpar.dlt_ekin, m_iVol*dltpar.dlt_pos.mag());
}
