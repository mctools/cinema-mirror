#include <vector>
#include "PTMath.hh"
#include "PTException.hh"

std::vector<double> Prompt::logspace(double start, double stop, unsigned num)
{
  pt_assert(num>1);
  pt_assert(stop>start);
  std::vector<double> vec(num) ;
  double interval = (stop-start)/(num-1);
  for(std::vector<double>::iterator it=vec.begin();it!=vec.end();++it)  {
    *it = std::pow(10.0,start);
    start += interval;
  }
  vec.back() = std::pow(10.0,stop);
  return vec;
}

std::vector<double>  Prompt::linspace(double start, double stop, unsigned num)
{
  pt_assert(num>1);
  pt_assert(stop>start);
  std::vector<double> v;
  v.reserve(num) ;
  const double interval = (stop-start)/(num-1);
  //Like this for highest numerical precision:
  for (unsigned i = 0; i<num;++i)
    v.push_back(start+i*interval);
  v.back() = stop;
  return v;
}
