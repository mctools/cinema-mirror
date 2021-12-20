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

#include <vector>
#include "PTMath.hh"
#include "PTException.hh"

std::vector<double> Prompt::logspace(double start, double stop, unsigned num)
{
  pt_assert(num>1);
  pt_assert(stop>start);
  std::vector<double> vec(num) ;
  double interval = (stop-start)/(num-1);
  for(std::vector<double>::iterator it=vec.begin();it!=vec.end();++it)  {
    *it = std::pow(10.0,start);
    start += interval;
  }
  vec.back() = std::pow(10.0,stop);
  return vec;
}

std::vector<double>  Prompt::linspace(double start, double stop, unsigned num)
{
  pt_assert(num>1);
  pt_assert(stop>start);
  std::vector<double> v;
  v.reserve(num) ;
  const double interval = (stop-start)/(num-1);
  //Like this for highest numerical precision:
  for (unsigned i = 0; i<num;++i)
    v.push_back(start+i*interval);
  v.back() = stop;
  return v;
}
