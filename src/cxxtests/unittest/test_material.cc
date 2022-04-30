////////////////////////////////////////////////////////////////////////////////
//                                                                            //
//  This file is part of Prompt (see https://gitlab.com/xxcai1/Prompt)        //
//                                                                            //
//  Copyright 2021-2022 Prompt developers                                     //
//                                                                            //
//  Licensed under the Apache License, Version 2.0 (the "License");           //
//  you may not use this file except in compliance with the License.          //
//  You may obtain a copy of the License at                                   //
//                                                                            //
//      http://www.apache.org/licenses/LICENSE-2.0                            //
//                                                                            //
//  Unless required by applicable law or agreed to in writing, software       //
//  distributed under the License is distributed on an "AS IS" BASIS,         //
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  //
//  See the License for the specific language governing permissions and       //
//  limitations under the License.                                            //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

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
