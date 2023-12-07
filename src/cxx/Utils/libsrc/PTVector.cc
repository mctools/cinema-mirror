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

#include "PTVector.hh"
#include <iostream>

std::ostream& Prompt::operator << (std::ostream &o, const Prompt::Vector& vec)
{
  return o<<"{ " << vec.x() <<", " << vec.y() << ", " << vec.z() << " }";
}

void Prompt::Vector::print() const
{
  std::cout << *this <<std::endl;
}

void Prompt::Vector::setMag(double f)
{
  if (f<0)
    PROMPT_THROW(BadInput,"NCVector::setMag(): Can't set negative magnitude.");
  double themag2 = mag2();
  if (!themag2)
    PROMPT_THROW(BadInput,"NCVector::setMag(): Can't scale null-vector.");
  double ff = f / sqrt(themag2);
  m_x *= ff;
  m_y *= ff;
  m_z *= ff;
}
