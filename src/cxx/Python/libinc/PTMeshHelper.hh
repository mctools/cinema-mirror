#ifndef Prompt_MeshHelper_hh
#define Prompt_MeshHelper_hh

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

void* pt_Transformation3D_new(void *consttrfm3Dobj);
void* pt_Transformation3D_newfromID(int id);
void pt_Transformation3D_delete(void *trfm3Dobj);
void pt_Transformation3D_multiple(void *trfm3Dobj1, void *trfm3Dobj2);
void pt_Transformation3D_transform(void *trfm3Dobj1, size_t numPt, double *in, double *out);
const char* pt_Transformation3D_print(void *trfm3Dobj);



size_t pt_countFullTreeNode();

const char* pt_getMeshName(size_t pvolID);
void pt_getLogVolumeInfo(size_t pvolID, char* cp);
void pt_meshInfo(size_t pvolID, size_t nSegments, size_t &npoints, size_t &nPlolygen, size_t &faceSize);
void pt_getMesh(size_t pvolID, size_t nSegments, float *points, size_t *NumPolygonPoints, size_t *faces);
void pt_printMesh();

#ifdef __cplusplus
}
#endif
#endif
