#include <cmath>
#include "Window.hh"
#include <iostream>
#include "Utils.hh"

void gauss(double alpha, unsigned M, double *window)
{
  for(unsigned i=0;i<M;++i)
  {
    double tmp=alpha*i/M*2;
    window[i]=exp(-0.5*tmp*tmp);
  }
}

void kaiser(double beta, unsigned M, double *window)
{
  double i_I0Beta = 1./ std::cyl_bessel_i(0., beta);
  double n = (M-1)/2.;
  double i_n=1./n;

  for(unsigned i=0;i<M/2;++i)
  {
    double temp=(i-n)*i_n;
    *(window+i) = std::cyl_bessel_i(0., beta*std::sqrt(1.-temp*temp))*i_I0Beta;
    *(window+M-1-i)=*(window+i);
  }
  if(M%2==1)
    *(window+M/2) = 1.;
}

void hanning(unsigned M, double *window)
{
  double i_M=1./(M-1.);
  for(unsigned i=0;i<M;++i)
    window[i] = 0.5*(1 - cos(2*M_PI*i*i_M));
}

void blackman(unsigned M, double *window)
{
  double i_M=1./M;
  for(unsigned i=0;i<M;++i)
    window[i] = 0.42-0.5*cos(2*M_PI*i*i_M)+0.08*cos(4*M_PI*i*i_M);
}

void hft(unsigned M, double *window,  unsigned order)
{
  double n=2.*M_PI/M;
  for(unsigned i=0;i<M;++i)
  {
    double x=i*n;
    window[i] = 1. - 1.985844164102*cos(x) + 1.791176438506 *cos(2*x)
    -1.282075284005*cos(3*x)+0.667777530266*cos(4*x)
    -.240160796576*cos(5*x)+0.056656381764*cos(6*x)
    -.008134974479*cos(7*x)+0.000624544650*cos(8*x)
    -.000019808998*cos(9*x)+0.000000132974*cos(10*x)
    ;
  }
}
