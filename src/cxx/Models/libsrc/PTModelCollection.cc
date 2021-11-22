#include "PTModelCollection.hh"
#include "PTNCrystalScat.hh"
#include "PTNCrystalAbs.hh"
#include "PTPhysicsModel.hh"

Prompt::ModelCollection::ModelCollection()
:m_cache({}), m_oriented(false), m_rng( Singleton<SingletonPTRand>::getInstance() )
{}

Prompt::ModelCollection::~ModelCollection() {}

void Prompt::ModelCollection::addPhysicsModel(const std::string &cfg)
{
  m_models.emplace_back(std::make_shared<NCrystalAbs>(cfg));
  m_models.emplace_back(std::make_shared<NCrystalScat>(cfg, 1.));
  if(m_models.back()->isOriented())
    m_oriented=true;
  m_cache.cache_xs.resize(m_models.size());
}

bool Prompt::ModelCollection::sameInquiryAsLastTime(double ekin, const Vector &dir) const
{
  return m_oriented ? (m_cache.ekin==ekin && m_cache.dir == dir) : m_cache.ekin==ekin;
}

double Prompt::ModelCollection::totalCrossSection(double ekin, const Vector &dir) const
{
  if(sameInquiryAsLastTime(ekin, dir))
  {
    return m_cache.tot;
  }
  else
  {
    double xs(0.);
    for(unsigned i=0;i<m_models.size();i++)
    {
      double channelxs = m_oriented ? m_models[i]->getCrossSection(ekin, dir) :
                                      m_models[i]->getCrossSection(ekin);
      m_cache.cache_xs[i] = channelxs;
      xs += channelxs;
    }
    m_cache.tot = xs;
    m_cache.ekin = ekin;
    m_cache.dir = dir;
    return xs;
  }
}

void Prompt::ModelCollection::sample(double ekin, const Vector &dir, double &final_ekin, Vector &final_dir, double &scaleWeight) const
{
  if(!sameInquiryAsLastTime(ekin, dir))
    printf("WARNING, sampling event with different incident energy and/or direction\n");

  if(!m_cache.tot)
  {
    final_ekin = ekin;
    final_dir = dir;
    return;
  }

  double r1 =  m_rng.generate();
  unsigned i=0;
  double p(0.), i_tot(1./m_cache.tot);
  for(; i<m_cache.cache_xs.size(); i++) //fixme: this is only faster when the number of physics model is small
  {
    p += m_cache.cache_xs[i]*i_tot;
    if(p > r1)
      break;
  }
  m_models[i]->generate(ekin, dir, final_ekin, final_dir, scaleWeight);
}
