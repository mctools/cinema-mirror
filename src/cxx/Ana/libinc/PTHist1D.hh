#ifndef Hist1D_hh
#define Hist1D_hh

#include "PTHistBase.hh"
#include <cmath>

namespace Prompt {
  class Hist1D : public HistBase {
  public:
    explicit Hist1D(double xmin, double xmax, unsigned nbins,bool linear=true);
    virtual ~Hist1D();

    unsigned dimension() const override { return 1; }  ;
    std::vector<double> getEdge() const;
    void save(const std::string &filename) const override;

    void fill(double val);
    void fill(double val, double weight);

  private:
    double m_binfactor;
    double m_logxmin;
    bool m_linear;
  };
}

#endif
