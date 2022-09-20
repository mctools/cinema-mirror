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

#include "PTProgressMonitor.hh"

Prompt::ProgressMonitor::ProgressMonitor(const std::string& name, double numTask, double interval)
:m_name(name), m_numTask(numTask), m_currentTask(0),
m_completedRatio(0), m_interval(interval), m_estimated_ms(0), m_begin(std::chrono::steady_clock::now())
{

}

Prompt::ProgressMonitor::~ProgressMonitor()
{
  printf("Progress Monitor \"%s\" Summary: recorded %.2e events per second.\n", m_name.c_str(), m_currentTask/m_estimated_ms*1e3);
}

void Prompt::ProgressMonitor::OneTaskCompleted()
{
  m_currentTask++;
  auto end = std::chrono::steady_clock::now();
  double elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - m_begin).count();
  double compltedRatio = m_currentTask/m_numTask;
  if(compltedRatio-m_completedRatio>m_interval)
  {
    m_completedRatio=compltedRatio;
    double estimateTotal = elapsedTime/compltedRatio;
    m_estimated_ms=estimateTotal;
    double left = estimateTotal-elapsedTime;
    printf("%s, estimated %gs, progress %g%%, remaining %gs. \n",
    m_name.c_str(), estimateTotal*0.001, compltedRatio*100, left*0.001);
  }
}
