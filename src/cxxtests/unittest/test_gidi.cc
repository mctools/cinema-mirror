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

#include <iostream>
#include <memory>
#include "PTGIDI.hh"
#include "PTNCrystalScat.hh"
#include "PTHist1D.hh"

namespace pt = Prompt;

TEST_CASE("GeoLoader")
{
  auto &fac = Prompt::Singleton<Prompt::GIDIFactory>::getInstance();  
  std::shared_ptr<Prompt::GIDIModel> gidimodel = fac.createGIDIModel("C12", 1.);
  double ekin(0.0253);

  std::cout << "Gidi xs "<<gidimodel->getCrossSection(ekin)/Prompt::Unit::barn << std::endl;

  pt::Vector in(1,0,0);
  pt::Vector out(0,0,0);
  double ekin_out;
  double gidiekin(0);
  int loop=10000000;
  auto hist1 = pt::Hist1D("gidi_hist", 0, 0.1, 100);
  for(int i=0;i<loop;i++)
  {
    gidimodel->generate(ekin, in, ekin_out, out );
    if(ekin_out!=-1.)
    gidiekin += ekin_out;
    hist1.fill(ekin_out);
  }
  std::cout << "gidiekin " << gidiekin/loop << std::endl;
  hist1.save("histgidi.mcpl");

  auto pc = Prompt::NCrystalScat( "freegas::C/1.225kgm3/C_is_1.00_C12;temp=299.397" );
  std::cout << "NCrystal xs " <<pc.getCrossSection(ekin)/Prompt::Unit::barn << std::endl;
  gidiekin = 0;

  auto hist2 = pt::Hist1D("ncrystal_hist", 0, 0.1, 100);

  for(int i=0;i<loop;i++)
  {
    pc.generate(ekin, in, ekin_out, out );
    gidiekin += ekin_out;
    hist2.fill(ekin_out);
  }
  std::cout << "gidiekin ncrystal " << gidiekin/loop << std::endl;
  hist2.save("histnc.mcpl");

}
