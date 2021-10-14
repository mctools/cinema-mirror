#include "Hist2D.hh"

Prompt::Hist2D::Hist2D(double xmin, double xmax, unsigned xnbins,
                       double ymin, double ymax, unsigned ynbins)
:HistBase(xnbins*ynbins), m_xbinfactor(xnbins/(xmax-xmin)),
m_ybinfactor(ynbins/(ymax-ymin))
{
  m_xmin=xmin, m_xmax=xmax, m_xnbins=xnbins;
  m_ymin=ymin, m_ymax=ymax, m_ynbins=ynbins;
  m_nbins = m_xnbins * m_ynbins;

  if(xnbins*ynbins==0)
    PROMPT_THROW(BadInput, "bin size is zero");

  if(xmax<=xmin || ymax<=ymin)
    PROMPT_THROW(BadInput, "max<min");

}

Prompt::Hist2D::~Hist2D()
{
}

void Prompt::Hist2D::operator+=(const Hist2D& hist)
{
  auto data=hist.getRaw();
  if(data.size()!=m_data.size())
    PROMPT_THROW(BadInput, "operator+= hist with different data size");
  std::lock_guard<std::mutex> guard(m_hist_mutex);
  for(unsigned i=0;i<data.size();++i)
    m_data[i]+=data[i];
}

void Prompt::Hist2D::save(const std::string &filename) const
{
  PROMPT_THROW(BadInput, "not yet implemented");
}

//Normal filling:
void Prompt::Hist2D::fill(double xval, double yval)
{
  fill(xval, yval, 1.);
}

void Prompt::Hist2D::fill(double xval, double yval, double w)
{
  std::lock_guard<std::mutex> guard(m_hist_mutex);
  fill_unguard(xval, yval, w);
}
