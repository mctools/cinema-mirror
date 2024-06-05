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

#include "PTScorer.hh"
#include "PTActiveVolume.hh"

Prompt::Scorer::Scorer(const std::string& name, ScorerType type, unsigned int pdg, int groupid) 
  : m_name(name), m_type(type), m_groupid(groupid), m_pdg(pdg), m_activeVolume(Singleton<ActiveVolume>::getInstance())  { };


int Prompt::Scorer::getVolumeGroupID()
{
  return m_activeVolume.getVolume()->GetCopyNo();
}
