#ifndef Prompt_PTPythonScorer_hh
#define Prompt_PTPythonScorer_hh

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

#include "PromptCore.hh"

#ifdef __cplusplus
extern "C" {
#endif


//ScorerDeposition
void* pt_ScorerDeposition_new(const char* name, double xmin, double xmax, unsigned nbins, unsigned pdg, int type, bool linear, int groupid);

void* pt_ScorerESpectrum_new(const char* name, bool scoreTransfer, double xmin, double xmax, unsigned nbins, unsigned pdg, int type, int groupid, bool linear);

void* pt_ScorerTOF_new(const char* name, double xmin, double xmax, unsigned nbins, unsigned pdg, int type, int groupid);

void* pt_ScorerWlSpectrum_new(const char* name, double xmin, double xmax, unsigned nxbins, unsigned int pdg, int type, int groupid);

void* pt_ScorerVolFluence_new(const char* name, double xmin, double xmax, unsigned nbins, double volme, unsigned pdg, int type, bool linear, int groupid);

void* pt_ScorerMultiScat_new(const char* name, double xmin, double xmax, unsigned nbins, unsigned pdg, int type, int groupid);

void pt_addMultiScatter(void* scatter, void* espScorer, int scatNumReq);

#ifdef __cplusplus
}
#endif

#endif
