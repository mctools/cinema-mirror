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

#include "PTPython.hh"
#include "PTModelCollection.hh"
#include "PTParticleProcess.hh"

void* pt_makeModelCollection(const char * cfg)
{
  auto *proc = new Prompt::ParticleProcess(cfg);
  return static_cast<void *>(proc);
}


void pt_deleteModelCollection(void* obj)
{
  delete static_cast<Prompt::ParticleProcess *>(obj);
}

double pt_ModelCollection_getxs(void* obj, int pdg, double ekin)
{
  Prompt::Vector dir;
  return static_cast<Prompt::ParticleProcess *>(obj)
      ->getModelCollection()->totalCrossSection(pdg, ekin, dir)/Prompt::Unit::barn;
}

double pt_ModelCollection_generate(void* obj, double ekin)
{
  Prompt::Vector dir;
  double final_ekin(0); 
  Prompt::Vector final_dir;

  static_cast<Prompt::ParticleProcess *>(obj)
      ->getModelCollection()->generate(ekin, dir, final_ekin, final_dir);
  return 0;
}
