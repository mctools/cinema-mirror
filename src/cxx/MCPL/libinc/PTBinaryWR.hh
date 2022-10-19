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

  enum class PromtRecodeType { MCPL, SCRSQ };

  // typedef struct {
  //   double ekin;            /* kinetic energy [MeV]             */
  //   double polarisation[3]; /* polarisation vector              */
  //   double position[3];     /* position [cm]                    */
  //   double direction[3];    /* momentum direction (unit vector) */
  //   double time;            /* time-stamp [millisecond]         */
  //   double weight;          /* weight or intensity              */
  //   int32_t pdgcode;    /* MC particle number from the Particle Data Group (2112=neutron, 22=gamma, ...)        */
  //   uint32_t userflags; /* User flags (if used, the file header should probably contain information about how). */
  // } mcpl_particle_t;

  struct ScorerSqRecord {
    double ekin;
    double q, qtrue, ekin_atbirth;
    double ekin_tof;
    double dummy1[3];
    double time;
    double weight;

    int32_t scatNum;
    uint32_t dummy3;
  };

  struct PromtRecord
  {
    PromtRecodeType type;
    union
    {
        mcpl_particle_t mcplParticle;
        ScorerSqRecord sqRecode;
    };
  };


  class BinaryWrite {
  public:
    BinaryWrite(const std::string &fn, bool enable_double=false, bool enable_extra3double=false, bool enable_extraUnsigned=false);
    virtual ~BinaryWrite();

    // Header
    void addHeaderComment(const std::string &comment);
    template <typename T>
    void addHeaderData(const std::string &dataname, const T *data,
                      const std::vector<uint64_t> &shape, NumpyWriter::NPDataType type);
    constexpr void closeHeader() { m_headerClosed=true; }
    const std::string& getFileName() { return m_filename; }

    // Particle list
    void record(const Particle &p);
    void record(const PromtRecord &p);

    // const std::string& getFileName() { return m_filename; }

  protected:
    std::string m_filename;
    mcpl_outfile_t m_file;
    mcpl_particle_t *m_particleInFile;
    bool m_headerClosed;
  };
}

template <typename T>
void Prompt::BinaryWrite::addHeaderData(const std::string &dataname, const T *data,
                  const std::vector<uint64_t> &shape, NumpyWriter::NPDataType type)
{

  if(m_headerClosed)
      PROMPT_THROW(LogicError, "addHeaderData can not operate on a file when the file header is closed ");

  uint64_t datasize = 1;
  for(auto v: shape)
    datasize *= v;

  std::string npdata;
  NumpyWriter::makeNumpyArrFromUChar(reinterpret_cast<const uint8_t*>(data), sizeof(T)*datasize, type, shape, npdata );

  // std::cout << "npdata type " << typeid(T).name()  << "\n " << npdata << std::endl;

   mcpl_hdr_add_data(m_file, dataname.c_str(), npdata.size(), npdata.c_str());
}

#endif
