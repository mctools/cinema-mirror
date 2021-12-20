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

#include "PTHistBase.hh"
#include <stdexcept>
#include <fstream>
//fixme:
Prompt::HistBase::HistBase(unsigned nbin)
: m_data(nbin,0.), m_hit(nbin,0.), m_xmin(0), m_xmax(0),
 m_sumW(0), m_underflow(0), m_overflow(0),m_nbins(0)
{

}

Prompt::HistBase::~HistBase()
{
}


void Prompt::HistBase::scale(double scalefact)
{
  std::lock_guard<std::mutex> guard(m_hist_mutex);

  for(unsigned i=0;i<m_nbins;i++)
    m_data[i] *= scalefact;

  m_sumW *= scalefact;
  m_underflow *= scalefact;
  m_overflow *= scalefact;

}

void Prompt::HistBase::reset()
{
  std::lock_guard<std::mutex> guard(m_hist_mutex);

  std::fill(m_data.begin(), m_data.begin()+m_nbins, 0.);
  m_sumW = 0.;
  m_underflow = 0.;
  m_overflow = 0.;
}
