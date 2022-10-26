#ifndef Prompt_RotatingObj_hh
#define Prompt_RotatingObj_hh

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
#include "PTDiscreteModel.hh"
#include <memory>

namespace Prompt {

  // The rational axis is parameterised as L(t) = A+tD, where A is a point on the
  // axis, t is a factor, D is the direction.
  // The closest line of point P to L is R,
  // then t0 = D.(P-A)/(D.D), R = P-L(t0)
  // In the case where D is a unit vector, and B = P-A
  // R = (1-D.B)B
  // the velocity at the point P is then 2\pi\omega R


  class RotatingObj  : public DiscreteModel {
  public:
    RotatingObj(const std::string &cfgstringAsName, const Vector &dir, const Vector &point, double rotFreq);
    virtual ~RotatingObj();

    virtual double getCrossSection(double ekin) const override;
    virtual double getCrossSection(double ekin, const Vector &dir) const override;
    virtual void generate(double ekin, const Vector &dir, double &final_ekin, Vector &final_dir) const override;

    Vector getLinearVelocity(const Vector &pos);


  private:
    const Vector m_dir, m_point;
    double m_angularfreq;
  };

}

inline Prompt::Vector Prompt::RotatingObj::getLinearVelocity(const Vector &pos)
{
  Vector B = pos - m_point;
  return (B-m_dir*(m_dir.dot(B)))*m_angularfreq;
}

#endif
