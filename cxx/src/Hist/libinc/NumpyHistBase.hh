#ifndef NumpyHistBase_hh
#define NumpyHistBase_hh

#include <string>
#include <cassert>
#include <vector>
#include <stdexcept>
#include <mutex>

class NumpyHistBase {
public:
  explicit NumpyHistBase(unsigned nbins);
  virtual ~NumpyHistBase();

  double getXMin() const {return m_xmin;}
  double getXMax() const {return m_xmax;}
  double getIntegral() const {return m_sumW;};
  double getOverflow() const {return m_overflow;};
  double getUnderflow() const {return m_underflow;};
  double getNBins() const {return m_nbins;};

  void scale(double scalefact);
  void reset();
  void save(const std::string &filename) const;
  const std::vector<double>& getRaw() const {return m_data;}

  virtual unsigned dimension() const = 0;
  virtual void serialise(std::string &serialised) const = 0;

protected:
  void serialise_numpy(const std::vector<uint64_t> &shape, std::string &serialised) const;
  mutable std::mutex m_hist_mutex;
  std::vector<double> m_data;
  //there is no function to modify following private mambers, so they are not const
  double m_xmin;
  double m_xmax;
  double m_sumW;
  double m_underflow;
  double m_overflow;
  double m_minfilled;
  double m_maxfilled;
  uint32_t m_nbins;

private:
  //Copy/assignment are forbidden to avoid troubles
  // Move initialization
  NumpyHistBase(NumpyHistBase&& other) = delete;
  // Copy initialization
  NumpyHistBase(const NumpyHistBase& other) = delete;
  // Move assignment
  NumpyHistBase& operator = (NumpyHistBase&& other) = delete;
  // Copy assignment
  NumpyHistBase& operator = (const NumpyHistBase& other) = delete;

};

#endif
