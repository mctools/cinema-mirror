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

#include "PTGidiSetting.hh"
   
Prompt::GidiSetting::GidiSetting()
: m_enablegidi(false), m_gidithreshold(5.), 
m_gidiSampleNonTransportingParticles(true), 
m_gidiEnableGamma(false),
m_gidiPowerIteration(false),
m_gidipops(getDataPath()+"pops.xml"), 
m_gidimap(getDataPath()+"all.map")
{
}


Prompt::GidiSetting::~GidiSetting()
{
}
