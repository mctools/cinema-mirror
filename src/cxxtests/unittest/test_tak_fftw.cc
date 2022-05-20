#include "../doctest.h"
#include "Fourier.hh"
#include <vector>
#include <iostream>
#include <iomanip> //setprecision
#include <omp.h>
#include <chrono>
#include "PTMath.hh"
namespace pt = Prompt;


TEST_CASE("fftw3")
{
  std::vector<double> input;
  for(unsigned i=0;i<5;i++)
  {
    input.push_back(i);
  }

  Fourier fr (10, true);
  std::vector<std::complex<double>> result;
  std::vector<double> resultreal;

  std::vector<std::complex<double>> result_ref {
            std::complex<double>(10,0),
            std::complex<double>(3.48614339197965,-8.56853559227205),
            std::complex<double>(-5.41421356237309,-4.82842712474619),
            std::complex<double>(-3.80316872754187,2.80995720221089),
            std::complex<double>(2,2),
            std::complex<double>(0.974741602795675,-2.36161567304292),
            std::complex<double>(-2.58578643762691,-0.82842712474619),
            std::complex<double>(-0.657716267233461,2.25989153247415),
            std::complex<double>(2,0),
            std::complex<double>(-0.657716267233461,-2.25989153247415),
            std::complex<double>(-2.58578643762691,0.82842712474619),
            std::complex<double>(0.974741602795675,2.36161567304292),
            std::complex<double>(2,-2),
            std::complex<double>(-3.80316872754187,-2.80995720221089),
            std::complex<double>(-5.41421356237309,4.82842712474619),
            std::complex<double>(3.48614339197965,8.56853559227205) };

  std::vector<double> resultreal_ref {100, 85.5729979454762, 52.6274169979695, 22.3599518484093, 8, 6.52734977938266,
    7.37258300203048, 5.53970042673186, 4 ,5.53970042673186, 7.37258300203048, 6.52734977938266, 8, 22.3599518484093,
    52.6274169979695, 85.5729979454762};

  //same as fft.fft(np.array([0,1,2,3,4]), n=16
  fr.c2c(input, result);

  for(unsigned i=0;i<result.size();++i)
  {
    CHECK(pt::floateq(result[i].real(), result_ref[i].real()));
    std::cout  << result[i].real() << " ...  " << result_ref[i].real() << " "  << std::endl ;

    CHECK(pt::floateq(result[i].imag(), result_ref[i].imag()));
    std::cout  << std::setprecision(15)  << result[i] << " " ;
  }
  std::cout << std::endl;

  fr.autoCorrSpectrum(input, resultreal);
  for(unsigned i=0;i<resultreal.size();++i)
  {
    CHECK(pt::floateq(resultreal[i],resultreal_ref[i]));
    std::cout << std::setprecision(15)  << resultreal[i] << " " ;
  }
  std::cout << std::endl;
}
