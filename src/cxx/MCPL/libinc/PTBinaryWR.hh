#ifndef Prompt_BinaryWR_hh
#define Prompt_BinaryWR_hh

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

#include "PromptCore.hh"
#include "PTParticle.hh"
#include "mcpl.h"

// struct mcpl_outfile_t;
// struct mcpl_particle_t;

namespace Prompt {

  class BinaryWrite {
  public:
    BinaryWrite(const std::string &fn, bool with_extra3double=false, bool with_extraUnsigned=false);
    virtual ~BinaryWrite() = default;
    void record(const Particle &p);

  protected:
    mcpl_outfile_t m_file;
    mcpl_particle_t *m_particleSpace;
    virtual void configHeaderAndData();
  };

}

#endif
