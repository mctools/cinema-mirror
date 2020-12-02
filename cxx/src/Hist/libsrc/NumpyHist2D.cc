#include "NumpyHist2D.hh"
#include <cmath>
#include <vector>
#include <iostream>

NumpyHist2D::NumpyHist2D(unsigned xnbins, double xmin, double xmax,
                         unsigned ynbins, double ymin, double ymax)
:NumpyHistBase(xnbins*ynbins), m_xbinfactor(xnbins/(xmax-xmin)),
m_ybinfactor(ynbins/(ymax-ymin))
{
  m_xmin=xmin, m_xmax=xmax, m_xnbins=xnbins;
  m_ymin=ymin, m_ymax=ymax, m_ynbins=ynbins;
  m_nbins = m_xnbins * m_ynbins;

  if(xnbins*ynbins==0)
     std::invalid_argument("bin size is zero");

  if(xmax<=xmin || ymax<=ymin)
    std::invalid_argument("max min");

}

NumpyHist2D::~NumpyHist2D()
{
}

void NumpyHist2D::serialise(std::string &serialised) const
{
  serialise_numpy(std::vector<uint64_t>{m_xnbins, m_ynbins}, serialised);
}

//Normal filling:
void NumpyHist2D::fill(double xval, double yval)
{
  fill(xval, yval, 1.);
}

void NumpyHist2D::fill(double xval, double yval, double w)
{
  const unsigned ix = floor((xval-m_xmin)*m_xbinfactor);
  const unsigned iy = floor((yval-m_ymin)*m_ybinfactor);

  std::lock_guard<std::mutex> guard(m_hist_mutex);
  m_sumW += w;
  if(xval<m_xmin ||  yval<m_ymin) {
    m_underflow+=w;
    return;
  }
  else if(xval>m_xmax || yval>m_ymax) {
    m_overflow+=w;
    return;
  }
  m_data[ix*m_ynbins + iy]+=w;
}
