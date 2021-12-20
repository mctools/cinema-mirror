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

#include <string>
#include "PTHistCtypes.hh"
#include "PTHist2D.hh"
#include "PTHist1D.hh"
#include "PTHistBase.hh"

//Hist1D
void* Hist1D_new(double xmin, double xmax, unsigned nxbins, bool log)
{
  return static_cast<void*>(new Prompt::Hist1D(xmin,  xmax, nxbins, log));
}

void Hist1D_delete(void* obj)
{
  delete static_cast<Prompt::Hist1D*>(obj);
}

void Hist1D_fill(void* obj, unsigned n, double* xval)
{
  auto hist = static_cast<Prompt::Hist1D*>(obj);
  for(unsigned i=0;i<n;i++)
    hist->fill(xval[i]);
}
void Hist1D_fillWeighted(void* obj,unsigned n, double* xval, double* w)
{
  auto hist = static_cast<Prompt::Hist1D*>(obj);
  for(unsigned i=0;i<n;i++)
    hist->fill(xval[i], w[i]);
}

unsigned Hist2D_getNBinX(void* obj)
{
  return static_cast<Prompt::Hist2D*>(obj)->getNBinX();
}

unsigned Hist2D_getNBinY(void* obj)
{
  return static_cast<Prompt::Hist2D*>(obj)->getNBinY();
}

void* Hist2D_new(double xmin, double xmax, unsigned nxbins,
             double ymin, double ymax, unsigned nybins)
{
  return static_cast<void*>(new Prompt::Hist2D(xmin,  xmax, nxbins,
               ymin,  ymax, nybins));
}

void Hist2D_delete(void* obj)
{
    delete static_cast<Prompt::Hist2D*>(obj);
}

void Hist2D_fill(void* obj, unsigned n, double* xval, double* yval)
{
  auto hist=static_cast<Prompt::Hist2D*>(obj);
  for(unsigned i=0;i<n;i++)
    hist->fill(xval[i], yval[i]);
}

void Hist2D_fillWeighted(void* obj, unsigned n, double* xval, double* yval, double* w)
{
  auto hist = static_cast<Prompt::Hist2D*>(obj);
  for(unsigned i=0;i<n;i++)
    hist->fill(xval[i], yval[i], w[i]);
}

void HistBase_save(void* obj, char *fn)
{
  static_cast<Prompt::HistBase*>(obj)->save(std::string(fn));
}

void HistBase_scale(void* obj, double factor)
{
  static_cast<Prompt::HistBase*>(obj)->scale(factor);
}

unsigned HistBase_getNBin(void* obj)
{
  return static_cast<Prompt::HistBase*>(obj)->getNBin();
}


const double* HistBase_getRaw(void* obj)
{
  return (static_cast<Prompt::HistBase*>(obj)->getRaw()).data();
}

const double* HistBase_getHit(void* obj)
{
  return (static_cast<Prompt::HistBase*>(obj)->getHit()).data();
}
