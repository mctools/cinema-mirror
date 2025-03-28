#ifndef Prompt_IdealElaScat_hh
#define Prompt_IdealElaScat_hh

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
#include "PTDiscreteModel.hh"

namespace Prompt {

  //IdealElaScat is in fact a scatterer of NCrystal
  //Physics model should be initialised from material

  class IdealElaScat  : public DiscreteModel {
  public:
    IdealElaScat(double xs_barn, double density_per_aa3, double bias);
    virtual ~IdealElaScat();

    virtual double getCrossSection(double ekin) const override;
    virtual double getCrossSection(double ekin, const Vector &dir) const override;
    virtual void generate(double ekin, const Vector &dir, double &final_ekin, Vector &final_dir) const override;
    double getNumberDensity() const { return m_density; }
  private:
    double m_xs;
    double m_density;

  };

}

#endif
