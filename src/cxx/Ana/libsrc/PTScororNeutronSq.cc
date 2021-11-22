#include "PTScororNeutronSq.hh"

Prompt::ScororNeutronSq::ScororNeutronSq(const Vector &samplePos, const Vector &refDir,
      double sourceSampleDist, double qmin, double qmax, unsigned numbin, bool kill, bool linear)
:Scoror("ScororNeutronSq", ENTRY), m_samplePos(samplePos), m_refDir(refDir),
m_sourceSampleDist(sourceSampleDist), m_hist(std::make_unique<Hist1D>(qmin, qmax, numbin, linear)),
m_kill(kill)
{}

Prompt::ScororNeutronSq::~ScororNeutronSq() {
  save("ScororNeutronSq.dat");
}

void Prompt::ScororNeutronSq::score(Prompt::Particle &particle)
{
  double angle_cos = particle.getDirection().angleCos(m_refDir);
  double dist = m_sourceSampleDist+(particle.getPosition()-m_samplePos).mag();
  double v = dist/particle.getTime();
  double ekin = 0.5*const_neutron_mass_evc2*v*v;
  //static approximation
  double q = neutronAngleCosine2Q(angle_cos, ekin, ekin);
  m_hist->fill(q, particle.getWeight());
  if(m_kill)
    particle.kill();
}

void Prompt::ScororNeutronSq::score(Prompt::Particle &particle, const DeltaParticle &dltpar)
{
  score(particle);
}

void Prompt::ScororNeutronSq::save(const std::string &fname)
{
  m_hist->save(fname);
}
