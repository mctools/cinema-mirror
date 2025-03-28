#ifndef Prompt_Python_hh
#define Prompt_Python_hh

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

#include "PromptCore.hh"

#ifdef __cplusplus
extern "C" {
#endif

// Converters
double pt_eKin2k(double ekin);
double pt_angleCosine2Q(double anglecosine, double enin_eV, double enout_eV);
double pt_wl2ekin( double wl);
double pt_ekin2wl( double ekin) ;
double pt_ekin2speed( double ekin);
double pt_speed2ekin( double v);

// random
double pt_rand_generate();

// Prompt::Launcher
void* pt_Launcher_getInstance();
void pt_Launcher_setSeed(void* obj, uint64_t seed);
void pt_Launcher_setGun(void* obj, void* objgun);
void pt_Launcher_loadGeometry(void* obj, const char* fileName);
size_t pt_Launcher_getTrajSize(void* obj);
void pt_Launcher_getTrajectory(void* obj, double *trj);
void pt_Launcher_go(void* obj, uint64_t numParticle, double printPrecent, bool recordTrj, bool timer);

// Prompt::Hist1D
void* pt_Hist1D_new(double xmin, double xmax, unsigned nbins, bool linear);
void pt_Hist1D_delete(void* obj);
void pt_Hist1D_getEdge(void* obj, double* edge);
void pt_Hist1D_getWeight(void* obj, double* w);
void pt_Hist1D_getHit(void* obj, double* h);
void pt_Hist1D_fill(void* obj, double val, double weight);
void pt_Hist1D_fillmany(void* obj, size_t n, double* val, double* weight);

// Prompt::Est1D
void* pt_Est1D_new(double xmin, double xmax, unsigned nbins, bool linear);
void pt_Est1D_delete(void* obj);
void pt_Est1D_fill(void* obj, double val, double weight, double error);
void pt_Est1D_fillmany(void* obj, size_t n, double* val, double* weight, double* error);

// Prompt::Hist2D
void* pt_Hist2D_new(double xmin, double xmax, unsigned nxbins,
                    double ymin, double ymax, unsigned nybins);
void pt_Hist2D_delete(void* obj);
void pt_Hist2D_getWeight(void* obj, double* w);
void pt_Hist2D_getDensity(void* obj, double* d);
void pt_Hist2D_getHit(void* obj, double* h);
void pt_Hist2D_fill(void* obj, double xval, double yval, double weight);
void pt_Hist2D_fillmany(void* obj, size_t n, double* xval, double* yval, double* weight);
void pt_Hist2D_merge(void* obj, void* obj2);

#ifdef __cplusplus
}
#endif

#endif
