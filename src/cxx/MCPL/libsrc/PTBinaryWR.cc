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

#include "mcpl.h"
#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include "PTBinaryWR.hh"


Prompt::BinaryWrite::BinaryWrite(const std::string &fn, bool with_extra3double, bool with_extraUnsigned)
:m_file(mcpl_create_outfile(fn.c_str())), m_particleSpace(nullptr), m_headerClosed(false)
{
    mcpl_hdr_set_srcname(m_file,"my_cool_program_name");
    mcpl_enable_doubleprec(m_file);
    if(with_extra3double)   mcpl_enable_polarisation(m_file);  // double[3]
    if(with_extraUnsigned)  mcpl_enable_userflags(m_file);    // uint32_t

    mcpl_hdr_set_srcname(m_file, ("Prompt " + PTVersion).c_str());
    m_particleSpace = mcpl_get_empty_particle(m_file);
}

Prompt::BinaryWrite::~BinaryWrite()
{
  mcpl_closeandgzip_outfile(m_file);
}

void Prompt::BinaryWrite::addHeaderComment(const std::string &comment)
{
  if(m_headerClosed)
      PROMPT_THROW(LogicError, "addHeaderComment can not operate on a file when the file header is closed ");

  mcpl_hdr_add_comment(m_file, comment.c_str());
}

void Prompt::BinaryWrite::record(const Particle &p)
{
  m_particleSpace->pdgcode = p.getPGD();

  //fixme: position in centimeters:
  const Vector &pos = p.getPosition();
  m_particleSpace->position[0] = pos.x();
  m_particleSpace->position[1] = pos.y();
  m_particleSpace->position[2] = pos.z();

  //fixme: kinetic energy in MeV:
  m_particleSpace->ekin = p.getEKin();

  const Vector &dir = p.getDirection();

  m_particleSpace->direction[0] = dir.x();
  m_particleSpace->direction[1] = dir.y();
  m_particleSpace->direction[2] = dir.z();

  //time in milliseconds:
  m_particleSpace->time = p.getTime();

  //weight in unspecified units:
  m_particleSpace->weight = p.getWeight();

  //modify userflags (unsigned_32) and polarisation (double[3]) as well, if enabled.

  mcpl_add_particle(m_file, m_particleSpace);
}
