#include "NumpyHistBase.hh"
#include "RedisNumpy.hh"
#include <stdexcept>
#include <fstream>
//fixme: mutex dead lock!!!
NumpyHistBase::NumpyHistBase(unsigned nbin)
: m_data(nbin,0.), m_xmin(0), m_xmax(0),
 m_sumW(0), m_underflow(0), m_overflow(0),m_nbins(0)
{

}

NumpyHistBase::~NumpyHistBase()
{
}

void NumpyHistBase::save(const std::string &filename) const
{
  std::string bdata;
  serialise(bdata);
  std::ofstream outfile(filename, std::ofstream::binary);
  outfile << bdata;
  outfile.close();
}


void NumpyHistBase::serialise_numpy(const std::vector<uint64_t> &shape, std::string &serialise) const
{
  uint64_t elementnum=1;
  for(auto v:shape)
    elementnum *= v;

  if(elementnum!=m_nbins)
    std::invalid_argument("data can not be shaped by the given shape vector");

  RedisNumpy nvt;
  nvt.makeNumpyArr(m_data, RedisNumpy::data_type::f8,
                   shape, serialise);
}

void NumpyHistBase::scale(double scalefact)
{
  std::lock_guard<std::mutex> guard(m_hist_mutex);

  for(unsigned i=0;i<m_nbins;i++)
    m_data[i] *= scalefact;

  m_sumW *= scalefact;
  m_underflow *= scalefact;
  m_overflow *= scalefact;

}

void NumpyHistBase::reset()
{
  std::lock_guard<std::mutex> guard(m_hist_mutex);

  std::fill(m_data.begin(), m_data.begin()+m_nbins, 0.);
  m_sumW = 0.;
  m_underflow = 0.;
  m_overflow = 0.;
}
