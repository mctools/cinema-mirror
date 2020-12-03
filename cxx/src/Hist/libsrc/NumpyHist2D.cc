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
