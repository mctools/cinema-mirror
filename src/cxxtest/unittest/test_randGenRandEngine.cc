#include <chrono>
#include <iostream>

#include "PTLookUpTable.hh"
#include "PTRandCanonical.hh"
#include "PTRandEngine.hh"
#include "PTMath.hh"
namespace pt = Prompt;

#include "../doctest.h"
TEST_CASE("RandEngine")
{
  double upper=100;
  auto datax = pt::linspace(0,upper,100);
  auto datay = pt::linspace(0,upper,100);

  auto gen = std::make_shared<pt::RandEngine>(6402);
  auto RandCanonical = pt::RandCanonical<pt::RandEngine>(gen);
  auto lut = std::make_unique<pt::LookUpTable>(datax,datay);

  auto start = std::chrono::steady_clock::now();
  uint64_t loop = 1000000;
  double max(0),sum(0);

  for(uint64_t i=0;i<loop;i++)
  {
    double rdn =RandCanonical.generate();
    double x = rdn*upper;
    double y = lut->get(x);
    CHECK(pt::floateq(x,y));
    max=max<y?y:max;
    sum += rdn;
  }
  auto end = std::chrono::steady_clock::now();
  double nanos = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  printf("max %.16f\n",max);
  printf("sum %.16f\n",sum);
  CHECK(pt::floateq(sum,499722.3119361777207814));
  CHECK(pt::floateq(max,99.9999958758698568));

  std::cout << "Elapsed time in milliseconds : "
    << nanos/1e6 << "ms , " << nanos/loop << " ns per iteration"
    << " " << std::endl;
}
