#ifndef Hist2D_hh
#define Hist2D_hh

#include "HistBase.hh"
#include <cmath>

namespace Prompt {

  class Hist2D : public HistBase {
  public:

    explicit Hist2D(double xmin, double xmax, unsigned nxbins,
                 double ymin, double ymax, unsigned nybins);
    virtual ~Hist2D();

    void operator+=(const Hist2D& hist);
    unsigned dimension() const override { return 2; }  ;
    void save(const std::string &filename) const override;

    uint32_t getNBinX() const {return m_xnbins;}
    uint32_t getNBinY() const {return m_ynbins;}
    double getYMin() const {return m_ymin;}
    double getYMax() const {return m_ymax;}

    void fill(double xval, double yval);
    void fill(double xval, double yval, double weight);
    void fill_unguard(double xval, double yval, double weight);
    void fill_unguard(double xval, const std::vector<double>& yval, const std::vector<double>& weight);
    void fill_unguard(const std::vector<double>& xval, const std::vector<double>& yval, const std::vector<double>& weight);

  private:
    //there is no function to modify private mambers, so they are not const
    double m_xbinfactor, m_ybinfactor;
    double m_ymin;
    double m_ymax;
    uint32_t m_xnbins, m_ynbins;
  };
  #include "Hist2D.icc"
}



#endif
