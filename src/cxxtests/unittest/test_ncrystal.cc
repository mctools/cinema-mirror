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
#include "PTNCrystalScat.hh"
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

  auto pc = Prompt::NCrystalScat( "Al_sg225.ncmat;dcutoff=0.5;temp=25C" );
  double xs = pc.getCrossSection(1);
  Prompt::Vector out;
  double final(0);

  std::cout << xs << std::endl;
  printf("%.15f\n", xs);

  CHECK(Prompt::floateq(1.378536096609809*Prompt::Unit::barn, xs ));

  pc.generate(1., {1,0,0}, final, out);
  std::cout << final << " " << out << std::endl;

  pc.generate(1., {1,0,0}, final, out);
  std::cout << final << " " << out << std::endl;

  pc.generate(1., {1,0,0}, final, out);
  std::cout << final << " " << out << std::endl;


  //
  NCrystal::MatCfg gecfg("Ge_sg227.ncmat;dcutoff=0.5;mos=40arcsec;dir1=@crys_hkl:5,1,1@lab:0,0,1;dir2=@crys_hkl:0,-1,1@lab:0,1,0" );
  auto ge = NCrystal::createScatter(gecfg);

  auto wl = NCrystal::NeutronWavelength{1.540};
  auto xsect = ge.crossSection( wl, { 0.0, 1.0, 1.0 } );
  std::cout << "single crystal Ge x-sect at "<<wl<<" Aa is "<<xsect<<" barn (orientation 1)"<<std::endl;


  //test info
  NCrystal::MatCfg matcfg("solid::B4C/2500gcm3/B_is_0.95_B10_0.05_B11" );
  auto info = NCrystal::createInfo(matcfg);
  const NCrystal::Info::Composition & comp = info->getComposition();

  for(const NCrystal::Info::CompositionEntry &v : comp)
  {
    double frac = v.fraction;
    const auto& atom = v.atom.data();
    std::cout << atom.elementName() << ": A " << atom.A() << ", Z " << atom.Z() << ", fraction " << frac << std::endl;
    std::cout << atom.isComposite() << " " << atom.isNaturalElement() << " "  << std::endl;   
    if(atom.isComposite())
    {
      // for (unsigned i=0;i<atom.nComponents();i++)
      // {
      //   auto &com = atom.getComponent(i);
      //   std::cout <<  ": A " << com.A() << ", Z " << com.Z() << ", fraction " << frac << std::endl;

      //   std::cout << "Component " << i << com.data()  << std::endl;
      // }      
    }
  }

}
