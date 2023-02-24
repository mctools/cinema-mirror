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

#include "PTPython.hh"
#include "PTLauncher.hh"
#include "NCrystal/NCrystal.hh"

namespace pt = Prompt;

class SingletonPTRandWrapper : public NCrystal::RNGStream{
public:
  SingletonPTRandWrapper()
  :NCrystal::RNGStream(), m_ptrng(Prompt::Singleton<Prompt::SingletonPTRand>::getInstance())
  {}
  virtual ~SingletonPTRandWrapper() override {}

  double actualGenerate() override {return m_ptrng.generate(); }

  //For the sake of example, we wrongly claim that this generator is safe and
  //sensible to use multithreaded (see NCRNG.hh for how to correctly deal with
  //MT safety, RNG states, etc.):
  bool useInAllThreads() const override { return true; }
private:
  Prompt::SingletonPTRand &m_ptrng;
};

void pt_enable_prompt()
{
  //This checks that the included NCrystal headers and the linked NCrystal
  //library are from the same release of NCrystal:
  NCrystal::libClashDetect();

  //set the generator for ncrystal
  NCrystal::setDefaultRNG(NCrystal::makeSO<SingletonPTRandWrapper>());

}

double pt_rand_generate()
{
  return pt::Singleton<pt::SingletonPTRand>::getInstance().generate();
}

void* pt_Launcher_getInstance()
{
  return static_cast<void *>(std::addressof(pt::Singleton<pt::Launcher>::getInstance()));
}

void pt_Launcher_setSeed(void* obj, uint64_t seed)
{
  static_cast<pt::Launcher *>(obj)->setSeed(seed);
}

void pt_Launcher_setGun(void* obj, void* objgun)
{

}

void pt_Launcher_loadGeometry(void* obj, const char* fileName)
{
  static_cast<pt::Launcher *>(obj)->loadGeometry(std::string(fileName));
}

void pt_Launcher_steupFakeGeoPhyisc(void* obj)
{
  static_cast<pt::Launcher *>(obj)->steupFakeGeoPhyisc();
}

size_t pt_Launcher_getTrajSize(void* obj)
{
  return static_cast<pt::Launcher *>(obj)->getTrajSize();
}

void pt_Launcher_getTrajectory(void* obj, double *trj)
{
  auto &trjv = static_cast<pt::Launcher *>(obj)->getTrajectory();
  for(const auto &v: trjv)
  {
    *(trj++) = v.x();
    *(trj++) = v.y();
    *(trj++) = v.z();
  }
}

void pt_Launcher_go(void* obj, uint64_t numParticle, double printPrecent, bool recordTrj, bool timer)
{
  static_cast<pt::Launcher *>(obj)->go(numParticle, printPrecent, recordTrj, timer);
}
