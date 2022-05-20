#include "../doctest.h"
#include "Window.hh"
#include <vector>
#include <iostream>
#include <iomanip> //setprecision

#include "PTMath.hh"

namespace pt = Prompt;

TEST_CASE("window")
{
  // reference from numpy.kaiser(9,12.)
  std::vector<double> ref{5.27734413e-05, 2.12783766e-02, 2.15672745e-01, 6.94516009e-01,
       1.00000000e+00, 6.94516009e-01, 2.15672745e-01, 2.12783766e-02,
       5.27734413e-05};
  std::vector<double> window(9,0);
  kaiser(12, window.size(), window.data());
  for(unsigned i=0;i<window.size();++i)
  {
    std::cout  << window[i] << " "  << std::endl ;
    CHECK(pt::floateq(window[i], ref[i], 1e-9, 1e-9 ));

  }
}
