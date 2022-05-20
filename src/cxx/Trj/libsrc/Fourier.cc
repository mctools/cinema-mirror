#include "Fourier.hh"
#include <algorithm> //fill, swap_ranges
#include <cstring> //memcpy
#include "Utils.hh"
#include <iostream>


Fourier::Fourier(unsigned fftSize, bool forward)
:m_fftSize(nextGoodFFTNumber(fftSize)), m_forward(forward)
{
  std::lock_guard<std::mutex> guard(m_planMutex);

  m_buffer = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * m_fftSize);
  m_bufferAsComPlx = reinterpret_cast<std::complex<double>* > (m_buffer);

  //in-place transformation
  m_plan = forward ? fftw_plan_dft_1d(m_fftSize, m_buffer, m_buffer, FFTW_FORWARD, FFTW_MEASURE):
           fftw_plan_dft_1d(m_fftSize, m_buffer, m_buffer, FFTW_BACKWARD, FFTW_MEASURE);

}

Fourier::~Fourier()
{
  std::lock_guard<std::mutex> guard(m_planMutex);
  fftw_destroy_plan(m_plan);
  fftw_free(m_buffer);
}

void Fourier::c2c(const std::vector<double> &data, std::vector<std::complex<double>> &out) const
{
  std::vector<std::complex<double>> in;
  in.assign(data.data(), data.data()+data.size());
  c2c(in,out);
}

void Fourier::c2c(const std::vector<std::complex<double>> &data, std::vector<std::complex<double>> &out) const
{
  unsigned n=data.size();
  memcpy( m_buffer, data.data(), sizeof( fftw_complex )*n );
  processBuffer(n, out);
}

void Fourier::autoCorrSpectrum(const std::vector<double> &data,
                std::vector<double> &out) const
{
  std::vector<std::complex<double>> in;
  in.assign(data.data(), data.data()+data.size());
  autoCorrSpectrum(in, out);
}


void Fourier::autoCorrSpectrum(const std::vector<std::complex<double>> &data,
              std::vector<double> &results) const
{
  //As suggested in G.R. Kneller Comp. Phys. Comm., 1995,
  //input is padded with zero to double the size

  if(data.size()*2>m_fftSize)
  {
    throw std::runtime_error ("Fourier: input size *2 is greater than m_fftSize");
  }

  if(!m_forward)
    throw std::runtime_error ("Fourier: autoCorrelation must use forward fft");
  std::vector<std::complex<double>> out;
  c2c(data, out);


  results.resize(m_fftSize);
  for(unsigned j=0;j<m_fftSize;j++)
  {
    results[j] = out[j].real()*out[j].real()+out[j].imag()*out[j].imag();
  }
}

void Fourier::processBuffer(unsigned inputSize, std::vector<std::complex<double>> &out) const
{
  if(m_fftSize-inputSize)
  {
    std::fill(m_bufferAsComPlx+inputSize, m_bufferAsComPlx+m_fftSize, std::complex<double>(0,0));
  }
  {
    std::lock_guard<std::mutex> guard(m_planMutex);
    fftw_execute(m_plan);
  }
  out.resize(m_fftSize);
  memcpy( out.data(), m_buffer, sizeof( fftw_complex )*m_fftSize );
}
