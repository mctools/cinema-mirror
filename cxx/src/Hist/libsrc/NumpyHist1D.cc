#include "NumpyHist1D.hh"
#include <vector>

NumpyHist1D::NumpyHist1D(unsigned nbins, double xmin, double xmax)
:NumpyHistBase(nbins), m_binfactor(nbins/(xmax-xmin))
{
  m_xmin=xmin, m_xmax=xmax, m_nbins=nbins;
}

NumpyHist1D::~NumpyHist1D()
{
}


void NumpyHist1D::serialise(std::string &serialised) const
{
  serialise_numpy(std::vector<uint64_t>{m_nbins}, serialised);
}

//Normal filling:
void NumpyHist1D::fill(double val)
{
  std::lock_guard<std::mutex> guard(m_hist_mutex);

  if(val<m_xmin) {
    m_underflow++;
    return;
  }
  else if(val>m_xmax) {
    m_overflow++;
    return;
  }

  m_sumW++;

  fill_unguard(val);
}

void NumpyHist1D::fill(double val, double w)
{
  m_sumW+=w;
  std::lock_guard<std::mutex> guard(m_hist_mutex);

  if(val<m_xmin) {
    m_underflow += w;
    return;
  }
  else if(val>m_xmax) {
    m_overflow += w;
    return;
  }

  unsigned i = floor((val-m_xmin)*m_binfactor);
  m_data[i] += w;
}
