#include "../doctest.h"
#include "PTNCrystal.hh"
#include "PTMath.hh"
#include "PTModelCollection.hh"

TEST_CASE("ModelCollection")
{

  auto collection = Prompt::ModelCollection() ;
  collection.addPhysicsModel("Al_sg225.ncmat;dcutoff=0.5;temp=25C");

  double xs(0.);
  xs = collection.totalCrossSection(1., {0,0,0} );
  Prompt::Vector out;
  double final;
  std::cout << xs << std::endl;
  printf("%.15f\n", xs);

  CHECK(Prompt::floateq(1.378536096609809, xs ));

  collection.sample(1., {1,0,0}, final, out);
  std::cout << final << " " << out << std::endl;

  collection.sample(1., {1,0,0}, final, out);
  std::cout << final << " " << out << std::endl;

  collection.sample(1., {1,0,0}, final, out);
  std::cout << final << " " << out << std::endl;
}
