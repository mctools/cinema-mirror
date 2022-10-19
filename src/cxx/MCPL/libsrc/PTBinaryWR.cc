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


Prompt::BinaryWrite::BinaryWrite(const std::string &fn, bool enable_double, bool enable_extra3double, bool enable_extraUnsigned)
:m_filename(fn), m_file(mcpl_create_outfile(fn.c_str())), m_particleInFile(nullptr), m_enable_double(enable_double),
 m_enable_extra3double(enable_extra3double), m_enable_extraUnsigned(enable_extraUnsigned),
m_fileNotCreated(true), m_headerClosed(false) { }

void Prompt::BinaryWrite::init()
{
  m_fileNotCreated = false;
  mcpl_hdr_set_srcname(m_file,"Prompt");
  if(m_enable_double) mcpl_enable_doubleprec(m_file);
  if(m_enable_extra3double)   mcpl_enable_polarisation(m_file);  // double[3]
  if(m_enable_extraUnsigned)  mcpl_enable_userflags(m_file);    // uint32_t

  mcpl_hdr_set_srcname(m_file, ("Prompt " + PTVersion).c_str());
  m_particleInFile = mcpl_get_empty_particle(m_file);
}


Prompt::BinaryWrite::~BinaryWrite()
{
  if(!m_fileNotCreated)
    mcpl_closeandgzip_outfile(m_file);
}

void Prompt::BinaryWrite::addHeaderComment(const std::string &comment)
{
  if(m_fileNotCreated) init();
  if(m_headerClosed)
      PROMPT_THROW(LogicError, "addHeaderComment can not operate on a file when the file header is closed ");

  mcpl_hdr_add_comment(m_file, comment.c_str());
}

void Prompt::BinaryWrite::record(const PromptRecord &p)
{
  if(m_fileNotCreated) init();
  m_headerClosed=true;

  //size 12 double, a 32-bit int and a 32-bit unsigned
  // 8*12+4*2=104
  memcpy ( m_particleInFile, &(p.mcplParticle), sizeof(*m_particleInFile));
  mcpl_add_particle(m_file, m_particleInFile);
}

void Prompt::BinaryWrite::record(const Particle &p)
{
  if(m_fileNotCreated) init();

  m_headerClosed=true;
  m_particleInFile->pdgcode = p.getPGD();

  //fixme: position in centimeters:
  const Vector &pos = p.getPosition();
  m_particleInFile->position[0] = pos.x();
  m_particleInFile->position[1] = pos.y();
  m_particleInFile->position[2] = pos.z();

  //fixme: kinetic energy in MeV:
  m_particleInFile->ekin = p.getEKin();

  const Vector &dir = p.getDirection();

  m_particleInFile->direction[0] = dir.x();
  m_particleInFile->direction[1] = dir.y();
  m_particleInFile->direction[2] = dir.z();

  //time in milliseconds:
  m_particleInFile->time = p.getTime();

  //weight in unspecified units:
  m_particleInFile->weight = p.getWeight();

  //modify userflags (unsigned_32) and polarisation (double[3]) as well, if enabled.

  mcpl_add_particle(m_file, m_particleInFile);
}
