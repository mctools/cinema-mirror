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
#include "PTHist1D.hh"
#include "PTHist2D.hh"
#include "PTEst1D.hh"


namespace pt = Prompt;


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

void pt_Hist1D_getHit(void* obj, double* h)
{
  auto hit = static_cast<pt::Hist1D *>(obj)->getHit();
  for(size_t i=0;i<hit.size();i++)
  {
    h[i] = hit[i];
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


// Prompt::Hist2D
void* pt_Hist2D_new(double xmin, double xmax, unsigned nxbins,
                    double ymin, double ymax, unsigned nybins)
{
  return static_cast<void *>(new pt::Hist2D(xmin, xmax, nxbins,
                                            ymin, ymax, nybins));
}

void pt_Hist2D_delete(void* obj)
{
  delete static_cast<pt::Hist2D *>(obj);
}

void pt_Hist2D_getWeight(void* obj, double* w)
{
  auto weight = static_cast<pt::Hist2D *>(obj)->getRaw();
  for(size_t i=0;i<weight.size();i++)
  {
    w[i] = weight[i];
  }
}

void pt_Hist2D_merge(void* obj, void* obj2)
{
  auto hist1 = static_cast<Prompt::Hist2D*>(obj);
  hist1->merge(*static_cast<Prompt::HistBase*>(obj2));
}

void pt_Hist2D_fill(void* obj, double xval, double yval, double weight)
{
  static_cast<pt::Hist2D *>(obj)->fill(xval, yval, weight);
}

void pt_Hist2D_fillmany(void* obj, size_t n, double* xval, double* yval, double* weight)
{
  for(size_t i=0;i<n;i++)
    static_cast<pt::Hist2D *>(obj)->fill(xval[i], yval[i], weight[i]);
}

void pt_Hist2D_getHit(void* obj, double* h)
{
  auto hit = static_cast<pt::Hist2D *>(obj)->getHit();
  for(size_t i=0;i<hit.size();i++)
  {
    h[i] = hit[i];
  }
}


void pt_Hist2D_getDensity(void* obj, double* d)
{
  auto nbin = static_cast<pt::Hist2D *>(obj)->getNBin();
  auto weight = static_cast<pt::Hist2D *>(obj)->getRaw();
  auto hit = static_cast<pt::Hist2D *>(obj)->getHit();
  for(size_t i=0;i<nbin;i++)
  {
    if(hit[i])
      d[i]=weight[i]/hit[i];
  }
}

// Prompt::Hist1D
void* pt_Est1D_new(double xmin, double xmax, unsigned nbins, bool linear)
{
  return static_cast<void *>(new pt::Est1D(xmin, xmax, nbins, linear));
}

void pt_Est1D_delete(void* obj)
{
  delete static_cast<pt::Est1D *>(obj);
}

void pt_Est1D_fill(void* obj, double val, double weight, double error)
{
  static_cast<pt::Est1D *>(obj)->fill(val, weight, error);
}

void pt_Est1D_fillmany(void* obj, size_t n, double* val, double* weight, double* error)
{
  auto o = static_cast<pt::Est1D *>(obj);
  for(size_t i=0;i<n;i++)
  {
    o->fill(val[i], weight[i], error[i]);
  }
}
