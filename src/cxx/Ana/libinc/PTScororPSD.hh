#ifndef Prompt_ScororPSD_hh
#define Prompt_ScororPSD_hh

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
#include "PTScoror.hh"

namespace Prompt {

  class ScororPSD  : public Scoror2D {
  public:
    enum ScororType {XY, XZ, YZ};
  public:
    ScororPSD(const std::string &name, double xmin, double xmax, unsigned nxbins,
      double ymin, double ymax, unsigned nybins, ScororType type=XY);
    virtual ~ScororPSD();
    virtual void score(Particle &particle) override;
  private:
    ScororType m_type;
  };
}
#endif
