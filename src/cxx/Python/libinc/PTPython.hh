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

// random
double pt_rand_generate();

// Prompt::Launcher
void* pt_Launcher_getInstance();
void pt_Launcher_setSeed(void* obj, uint64_t seed);
void pt_Launcher_setGun(void* obj, void* objgun);
void pt_Launcher_loadGeometry(void* obj, const char* fileName);
size_t pt_Launcher_getTrajSize(void* obj);
void pt_Launcher_getTrajectory(void* obj, double *trj);
void pt_Launcher_go(void* obj, uint64_t numParticle, double printPrecent, bool recordTrj);

#ifdef __cplusplus
}
#endif

#endif
