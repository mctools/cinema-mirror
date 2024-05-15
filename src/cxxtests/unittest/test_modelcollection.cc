////////////////////////////////////////////////////////////////////////////////
//                                                                            //
//  This file is part of Prompt (see https://gitlab.com/xxcai1/Prompt)        //
//                                                                            //
//  Copyright 2021-2024 Prompt developers                                     //
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
#include "PTMath.hh"
#include "PTModelCollection.hh"
#include "PTParticleProcess.hh"

TEST_CASE("ModelCollection")
{
  std::vector<double> expectedEnergyOut;
  expectedEnergyOut = {0.0253, 0.0263, 0.045192303408606, 0.0283, -1, 0.00542322400666893 };
  std::vector<std::vector<double>> expectedDirOut;
  expectedDirOut ={
    { 0.211330297846913, -0.592201916881118, -0.777583689935858 },
    { -0.896705652516454, 0.227612286342529, -0.379620362798107 },
    { 0.349974039965383, -0.141170718428235, 0.926061012897511 },
    { -0.762662850218472, 0.547355775591302, -0.344596912092202 }, 
    { -0.762662850218472, 0.547355775591302, -0.344596912092202 }, //same as previous one due to absorption, variable outdir unchanged
    { -0.423603163886187, -0.904003071387581, 0.0577824062101281 }
  };
  double inE(0.0253);
  auto inDir = Prompt::Vector({1.,0.,0.});
  auto pp = Prompt::ParticleProcess("Al_sg225.ncmat");
  auto compModel = pp.getModelCollection();
  double xs(0.);
  std::cout.precision(15);
  std::cout << "Total XS: " << xs/Prompt::Unit::barn << std::endl;

  double finE(0.);
  Prompt::Vector finDir;
  for(size_t i;i<6;i++)
  {
    std::cout << "In energy: " << inE << std::endl;
    std::cout << "In dir: " << inDir << std::endl;
    xs = compModel->totalCrossSection(2112, inE, inDir);
    std::cout << "Total XS: " << xs/Prompt::Unit::barn << std::endl;
    compModel->generate(inE, inDir, finE, finDir);
    std::cout << "Out energy: " << finE << std::endl;
    CHECK(Prompt::floateq(finE, expectedEnergyOut[i]));
    std::cout << "Out dir: " << finDir << std::endl;
    CHECK(Prompt::floateq(finDir.x(), expectedDirOut[i][0]));
    CHECK(Prompt::floateq(finDir.y(), expectedDirOut[i][1]));
    CHECK(Prompt::floateq(finDir.z(), expectedDirOut[i][2]));
    std::cout << "  " << std::endl;
    inE += 0.001;
  }

  // auto collection = Prompt::ModelCollection(2112) ;
  // collection.addNCScaAbsModels("Al_sg225.ncmat;dcutoff=0.5;temp=25C");

  // double xs(0.);
  // xs = collection.totalCrossSection(1., {0,0,0} );
  // Prompt::Vector out;
  // double final;
  // std::cout << xs << std::endl;
  // printf("%.15f\n", xs);

  // CHECK(Prompt::floateq(1.378536096609809*Prompt::Unit::barn, xs ));


  // collection.generate(1., {1,0,0}, final, out);
  // std::cout << final << " " << out << std::endl;

  // collection.generate(1., {1,0,0}, final, out);
  // std::cout << final << " " << out << std::endl;

  // collection.generate(1., {1,0,0}, final, out);
  // std::cout << final << " " << out << std::endl;
}
