#ifndef NumpyHist1D_hh
#define NumpyHist1D_hh

#include "NumpyHistBase.hh"
#include <cmath>


class NumpyHist1D : public NumpyHistBase {
public:
  explicit NumpyHist1D(unsigned nbins, double xmin, double xmax);
  virtual ~NumpyHist1D();

  unsigned dimension() const override { return 1; }  ;
  void serialise(std::string &serialised) const override;

  void fill_unguard(double val);
  void fill(double val);
  void fill(double val, double weight);

private:
  double m_binfactor;
};


inline void NumpyHist1D::fill_unguard(double val)
{

  if(m_xmin < val && m_xmax > val)
  {
    unsigned i = floor((val-m_xmin)*m_binfactor);
    m_data[i]+=1;
  }
}


#endif
