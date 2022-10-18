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
#include "NumpyWriter.hh"

namespace Prompt {

  class BinaryWrite {
  public:
    BinaryWrite(const std::string &fn, bool with_extra3double=false, bool with_extraUnsigned=false);
    virtual ~BinaryWrite() = default;
    void record(const Particle &p);

    void addHeaderComment(const std::string &comment);
    template <typename T>
    void addHeaderData(const std::string &dataname, const T *data, size_t datasize, NumpyWriter::NPDataType type,
                      const std::vector<uint64_t> &shape);
    constexpr void closeHeader() { m_headerClosed=true; }

  protected:
    mcpl_outfile_t m_file;
    mcpl_particle_t *m_particleSpace;
    bool m_headerClosed;
  };
}

template <typename T>
void Prompt::BinaryWrite::addHeaderData(const std::string &dataname, const T *data, size_t datasize, NumpyWriter::NPDataType type,
                  const std::vector<uint64_t> &shape)
{
  if(m_headerClosed)
      PROMPT_THROW(LogicError, "addHeaderData can not operate on a file when the file header is closed ");

  uint64_t datasize2 = 1;
  for(auto v: shape)
    datasize2 *= v;

  if(datasize != datasize)
    PROMPT_THROW2(LogicError, "addHeaderData: the shape of the data: " << datasize2 <<", the given data size is " << datasize);

  std::string npdata;
  NumpyWriter::makeNumpyArrFromUChar(reinterpret_cast<const uint8_t*>(data), sizeof(T)*datasize, type, shape, npdata );

  mcpl_hdr_add_data(m_file, dataname.c_str(), sizeof(T)*datasize, npdata.c_str());
}

#endif
