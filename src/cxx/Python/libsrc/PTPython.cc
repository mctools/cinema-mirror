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
#include "PTHist1D.hh"

namespace pt = Prompt;


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

void pt_Launcher_go(void* obj, uint64_t numParticle, double printPrecent, bool recordTrj)
{
  static_cast<pt::Launcher *>(obj)->go(numParticle, printPrecent, recordTrj);
}


void* pt_Hist1D_new(double xmin, double xmax, unsigned nbins, bool linear)
{
    return static_cast<void *>(new pt::Hist1D(xmin, xmax, nbins, linear));
}

void pt_Hist1D_getEdge(void* obj, double* edge)
{
  auto edgevec = static_cast<pt::Hist1D *>(obj)->getEdge();
  for(size_t i=0;i<edgevec.size();i++)
  {
    edge[i]=edgevec[i];
  }
}

void pt_Hist1D_getWeight(void* obj, double* w)
{
  auto weight = static_cast<pt::Hist1D *>(obj)->getRaw();
  for(size_t i=0;i<weight.size();i++)
  {
    w[i] = weight[i];
  }
}

void pt_Hist1D_fill(void* obj, double val, double weight)
{
  static_cast<pt::Hist1D *>(obj)->fill(val, weight);
}

void pt_Hist1D_fillmany(void* obj, size_t n, double* val, double* weight)
{
  for(size_t i=0;i<n;i++)
    static_cast<pt::Hist1D *>(obj)->fill(val[i], weight[i]);
}

void pt_Hist1D_delete(void* obj)
{
  delete static_cast<pt::Hist1D *>(obj);
}
