#include "../doctest.h"
#include "Filon.hh"
#include <vector>
#include <iostream>
#include <iomanip> //setprecision
#include "PTMath.hh"

namespace pt = Prompt;

TEST_CASE("sin_integral_single")
{
  //floateq(0,1);

  std::vector<double> tp{1e-5,1e-3,1e-2,1e-1,1};
  int t_length=tp.size();
  double * sinRes = new double[t_length];
  double * calRes = new double[t_length];
  int len=11;
  double a=0.0;
  double b=1.0;
  double step=(b-a)/(len-1);
  double * omega = new double[len];
  double * fx = new double[len];
  for (auto i=0;i<len;i++)
  {
      omega[i]=0+step*i;
      fx[i]=1.0;
  }
  for (auto i=0; i<t_length; i++)
  {
      double t=tp[i];
      calRes[i]=1./t*(-cos(b*t)+cos(a*t));
      sin_integral_single((len-1)/2,omega,fx,t,sinRes[i]);
      std::cout << std::setprecision(15)<<sinRes[i]<<"||"<<calRes[i]<<std::endl;
      CHECK(pt::floateq(sinRes[i], calRes[i], 1e-12, 1e-12)); //fixme: the accuratcy of sin_integral_single is not as good as the cos_integral_single
  }

}

TEST_CASE("cos_integral_single")
{
  std::vector<double> tp{1e-5,1e-3,1e-2,1e-1,1};
  int t_length=tp.size();
  double * cosRes = new double[t_length];
  double * calRes = new double[t_length];
  int len=11;
  double a=0.0;
  double b=1.0;
  double step=(b-a)/(len-1);
  double * omega = new double[len];
  double * fx = new double[len];
  for (auto i=0;i<len;i++)
  {
      omega[i]=0+step*i;
      fx[i]=1.0;
  }
  for (auto i=0; i<t_length; i++)
  {
      double t=tp[i];
      calRes[i]=1./t*(sin(b*t)-sin(a*t));
      cos_integral_single((len-1)/2,omega,fx,t,cosRes[i]);
      std::cout<<cosRes[i]<<"||"<<calRes[i]<<std::endl;
      CHECK(pt::floateq(cosRes[i], calRes[i]));
  }
}
