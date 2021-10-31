#include "../doctest.h"

#include "PTMaxwellianGun.hh"
#include "PTNeutron.hh"
#include "PTHist1D.hh"

namespace pt = Prompt;

TEST_CASE("test_maxwellgun")
{
  auto hist = std::make_unique<pt::Hist1D>(0.0001, 0.3 , 100, false);
  auto gun = pt::MaxwellianGun(pt::Neutron(), 300, {1,1,-1400,1,1,0});
  for(unsigned i=0;i<10;i++)
  {
    auto p = gun.generate();
    std::cout << "event id " << p.getEventID()
    << " " << p.getPosition()
    << " " << p.getDirection()<< std::endl;
    hist->fill(p.getEKin());
  }
  std::cout << "integral " << hist->getIntegral() << std::endl;
  hist->save("test_maxwellgun");

}
