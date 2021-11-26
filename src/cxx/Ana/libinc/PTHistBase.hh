#ifndef HistBase_hh
#define HistBase_hh

#include <string>
#include <cassert>
#include <vector>
#include <stdexcept>
#include <mutex>
#include "PTException.hh"
#include "NumpyWriter.hh"

namespace Prompt {

  class HistBase {
  public:
    explicit HistBase(unsigned nbins);
    virtual ~HistBase();

    double getXMin() const {return m_xmin;}
    double getXMax() const {return m_xmax;}
    double getIntegral() const {return m_sumW;};
    double getOverflow() const {return m_overflow;};
    double getUnderflow() const {return m_underflow;};
    uint32_t getNBin() const {return m_nbins;};
    double getTotalHist() const {
      double sum(0);
      for(auto v: m_hit)
        sum += v;
      return sum;
    };

    void scale(double scalefact);
    void reset();

    const std::vector<double>& getRaw() const {return m_data;}
    const std::vector<double>& getHit() const {return m_hit;}

    virtual unsigned dimension() const = 0;
    virtual void save(const std::string &filename) const = 0;

  protected:

    mutable std::mutex m_hist_mutex;
    std::vector<double> m_data, m_hit;
    double m_xmin;
    double m_xmax;
    double m_sumW;
    double m_underflow;
    double m_overflow;
    uint32_t m_nbins;

  private:
    //Copy/assignment are forbidden to avoid troubles
    // Move initialization
    HistBase(HistBase&& other) = delete;
    // Copy initialization
    HistBase(const HistBase& other) = delete;
    // Move assignment
    HistBase& operator = (HistBase&& other) = delete;
    // Copy assignment
    HistBase& operator = (const HistBase& other) = delete;

  };
}

#endif
