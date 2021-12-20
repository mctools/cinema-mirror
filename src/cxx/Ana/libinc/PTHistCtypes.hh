#ifndef HistCtypes_hh
#define HistCtypes_hh

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

#ifdef __cplusplus
extern "C" {
#endif
  //HistBase
  void HistBase_save(void* obj, char *fn);
  void HistBase_scale(void* obj, double factor);
  const double* HistBase_getRaw(void* obj);
  const double* HistBase_getHit(void* obj);
  unsigned HistBase_getNBin(void* obj);


  //Hist1D
  void* Hist1D_new(double xmin, double xmax, unsigned nxbins, bool log);
  void Hist1D_delete(void* obj);
  void Hist1D_fill(void* obj, unsigned n, double* xval);
  void Hist1D_fillWeighted(void* obj,unsigned n, double* xval, double* weight);

  //Hist2D
  void* Hist2D_new(double xmin, double xmax, unsigned nxbins,
               double ymin, double ymax, unsigned nybins);
  void Hist2D_delete(void* obj);
  void Hist2D_fill(void* obj, unsigned n, double* xval, double* yval);
  void Hist2D_fillWeighted(void* obj,unsigned n, double* xval, double* yval, double* weight);
  unsigned Hist2D_getNBinX(void* obj);
  unsigned Hist2D_getNBinY(void* obj);

  #ifdef __cplusplus
  }
  #endif


#endif
