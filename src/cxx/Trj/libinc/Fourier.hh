#ifndef Fourier_hh
#define Fourier_hh

#include <complex>
#include "fftw3.h"
#include <mutex>
#include <vector>

class Fourier {
public:
  Fourier(unsigned fftSize, bool forward=true);
  virtual ~Fourier();

  unsigned getFFTSize() {return m_fftSize;}

  void c2c(const std::vector<double> &data, std::vector<std::complex<double>> &out) const;
  void c2c(const std::vector<std::complex<double>> &data, std::vector<std::complex<double>> &out) const;

  void autoCorrSpectrum(const std::vector<double> &data,
                  std::vector<double> &out) const;

  void autoCorrSpectrum(const std::vector<std::complex<double>> &data,
                  std::vector<double> &out) const;

  template <class T>
  void fftshift(std::vector<T> &data) const;

  template <class T>
  void positiveAxisOnly(std::vector<T> &data) const;


  // void c2c_expI(const std::vector<double> &data,  std::vector<std::complex<double>> &out) const;
  // void convolve_epxI(const std::vector<double> &data1, const std::vector<double> &data2,  std::vector<std::complex<double>> &out) const;

private:
  mutable std::mutex m_planMutex;
  const unsigned m_fftSize;
  mutable fftw_complex *m_buffer;
  mutable std::complex<double> *m_bufferAsComPlx;
  mutable fftw_plan m_plan;
  const bool m_forward;

  void processBuffer(unsigned inputSize, std::vector<std::complex<double>> &out) const;
};

template <class T>
void Fourier::fftshift(std::vector<T> &data) const
{
  size_t dataSize = data.size();
  if(dataSize%2)
    throw std::runtime_error ("Fourier: fftshift takes even number size input");

  size_t half=dataSize/2;
  std::swap_ranges(data.begin(), data.begin()+half, data.begin()+half);
}

template <class T>
void Fourier::positiveAxisOnly(std::vector<T> &data) const
{
  size_t dataSize = data.size();
  if(dataSize%2)
    throw std::runtime_error ("Fourier: positiveAxisOnly takes even number size input");

  size_t half=dataSize/2;
  data.resize(half);
}


// #ifdef __cplusplus
// extern "C" {
// #endif
// EXPORT_SYMBOL Fourier* Fourier_anew(unsigned fftSize, bool forward)
// {
//   return new Fourier(fftSize, forward);
// }
//
// EXPORT_SYMBOL void Fourier_delete(Fourier* self)
// {
//   delete self;
// }
//
// EXPORT_SYMBOL void Fourier_c2c(Fourier* self, const std::complex<double> *data, unsigned n, std::complex<double> *out, unsigned &outSize)
// {
//   self->c2c(data, n, out, outSize);
// }
//
// #ifdef __cplusplus
// }
// #endif

#endif
