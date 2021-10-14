#include "HistBase.hh"
#include <stdexcept>
#include <fstream>
//fixme:
Prompt::HistBase::HistBase(unsigned nbin)
: m_data(nbin,0.), m_hit(nbin,0.), m_xmin(0), m_xmax(0),
 m_sumW(0), m_underflow(0), m_overflow(0),m_nbins(0)
{

}

Prompt::HistBase::~HistBase()
{
}


void Prompt::HistBase::scale(double scalefact)
{
  std::lock_guard<std::mutex> guard(m_hist_mutex);

  for(unsigned i=0;i<m_nbins;i++)
    m_data[i] *= scalefact;

  m_sumW *= scalefact;
  m_underflow *= scalefact;
  m_overflow *= scalefact;

}

void Prompt::HistBase::reset()
{
  std::lock_guard<std::mutex> guard(m_hist_mutex);

  std::fill(m_data.begin(), m_data.begin()+m_nbins, 0.);
  m_sumW = 0.;
  m_underflow = 0.;
  m_overflow = 0.;
}
