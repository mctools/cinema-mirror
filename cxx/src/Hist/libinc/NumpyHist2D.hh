#ifndef NumpyHist2D_hh
#define NumpyHist2D_hh

#include "NumpyHistBase.hh"


class NumpyHist2D : public NumpyHistBase {
public:

  explicit NumpyHist2D(unsigned nxbins, double xmin, double xmax,
              unsigned nybins, double ymin, double ymax);
  virtual ~NumpyHist2D();

  unsigned dimension() const override { return 2; }  ;
  void serialise(std::string &serialised) const override;

  uint32_t getNBinX() const {return m_xnbins;}
  uint32_t getNBinY() const {return m_ynbins;}

  void fill(double xval, double yval);
  void fill(double xval, double yval, double weight);
  void filln(unsigned n, double *xval, double *yval);
  void filln(unsigned n, double *xval, double *yval, double *weight);
  void fill_unguard(double xval, double yval, double w);

private:
  //there is no function to modify private mambers, so they are not const
  double m_xbinfactor, m_ybinfactor;
  double m_ymin;
  double m_ymax;
  uint32_t m_xnbins, m_ynbins;


};

#include "NumpyHist2D.icc"
#endif
