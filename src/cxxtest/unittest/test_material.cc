#include "../doctest.h"
#include "PTMaterialPhysics.hh"

TEST_CASE("Material")
{
  auto mat = Prompt::MaterialPhysics() ;
  mat.addComposition("LiquidWaterH2O_T293.6K.ncmat", 1); //LiquidWaterH2O_T293.6K, Be_sg194, Al_sg225, UO2_sg225_UraniumDioxide
  double ekin  = 2e-3;
  double meanFreePath = 1/mat.macroCrossSection(ekin, {1,0,0} );
  double totlength(0.);
  unsigned loop(100000);
  for(unsigned i=0;i<loop;i++)
  {
    totlength += mat.sampleStepLength(ekin, {1,0,0});
  }
  printf("%.10f %.10f\n", meanFreePath, totlength/loop);
}
