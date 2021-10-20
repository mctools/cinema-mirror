#include "../doctest.h"
#include "PTNCrystal.hh"
#include "PTMath.hh"

TEST_CASE("NCrystal")
{

  // class CustomRNG : public NC::RNGStream {
  //   std::mt19937_64 m_gen;
  // protected:
  //   double actualGenerate() override { return NC::randUInt64ToFP01(static_cast<uint64_t>(m_gen())); }
  //   //For the sake of example, we wrongly claim that this generator is safe and
  //   //sensible to use multithreaded (see NCRNG.hh for how to correctly deal with
  //   //MT safety, RNG states, etc.):
  //   bool useInAllThreads() const override { return true; }
  // };
  //
  // //The NCrystal makeSO function is similar to std::make_shared
  // //and should be used instead of raw calls to new and delete:
  // auto rng = NC::makeSO<CustomRNG>();
  //
  // //Register:
  // NC::setDefaultRNG(rng);

  //////////////////////////////////////
  // Create and use aluminium powder: //
  //////////////////////////////////////

  auto pc = Prompt::PTNCrystal( "Al_sg225.ncmat;dcutoff=0.5;temp=25C" );
  double xs = pc.getCrossSection(1);
  Prompt::Vector out;
  double final;
  std::cout << xs << std::endl;
  printf("%.15f\n", xs);

  CHECK(Prompt::floateq(1.378536096609809*Prompt::Unit::barn, xs ));

  pc.generate(1., {1,0,0}, final, out);
  std::cout << final << " " << out << std::endl;

  pc.generate(1., {1,0,0}, final, out);
  std::cout << final << " " << out << std::endl;

  pc.generate(1., {1,0,0}, final, out);
  std::cout << final << " " << out << std::endl;
}
