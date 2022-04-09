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

#include "PTAnaManager.hh"
#include "PTUtils.hh"
#include "PTScororNeutronSq.hh"
#include "PTScororPSD.hh"
#include "PTScororVolFlux.hh"

Prompt::AnaManager::AnaManager()
{}

Prompt::AnaManager::~AnaManager()
{}


std::shared_ptr<Prompt::Scoror> Prompt::AnaManager::createScoror(const std::string &cfg, double vol)
{
  auto words = split(cfg, ';');
  std::cout << "Creating scoror with config: ";
  std::cout << cfg << "\n";
  //fixme check number of input config

  if(words[0]=="NeutronSq")
  {
    //type
    auto samplePos = string2vec(words[2]);
    auto neutronDir = string2vec(words[3]);
    double moderator2SampleDist = std::stod(words[4]);
    double minQ = std::stod(words[5]);
    double maxQ = std::stod(words[6]);
    int numBin = std::stoi(words[7]);
    if(words[8]=="ABSORB")
      return std::make_shared<Prompt::ScororNeutronSq>(words[1], samplePos, neutronDir, moderator2SampleDist, minQ, maxQ, numBin, Prompt::Scoror::ABSORB);
    else if(words[8]=="ENTRY")
      return std::make_shared<Prompt::ScororNeutronSq>(words[1], samplePos, neutronDir, moderator2SampleDist, minQ, maxQ, numBin, Prompt::Scoror::ENTRY);
    else
    {
      PROMPT_THROW2(BadInput, words[8] << " type is not supported by ScororNeutronSq");
      return std::make_shared<Prompt::ScororNeutronSq>(words[1], samplePos, neutronDir, moderator2SampleDist, minQ, maxQ, numBin, Prompt::Scoror::ENTRY);
    }
  }
  else if(words[0]=="PSD")
  {
    if(words[8]=="XY")
        return std::make_shared<ScororPSD>(words[1], std::stod(words[2]) , std::stod(words[3]) , std::stoi(words[4]) ,
                                          std::stod(words[5]) , std::stod(words[6]) , std::stoi(words[7]), ScororPSD::XY );
    else if(words[8]=="XZ")
        return std::make_shared<ScororPSD>(words[1], std::stod(words[2]) , std::stod(words[3]) , std::stoi(words[4]) ,
                                          std::stod(words[5]) , std::stod(words[6]) , std::stoi(words[7]), ScororPSD::XZ );
    else if(words[8]=="YZ")
        return std::make_shared<ScororPSD>(words[1], std::stod(words[2]) , std::stod(words[3]) , std::stoi(words[4]) ,
                                          std::stod(words[5]) , std::stod(words[6]) , std::stoi(words[7]), ScororPSD::YZ );
    else
    {
      PROMPT_THROW2(BadInput, words[8] << " type is not supported by ScororPSD");
      return std::make_shared<ScororPSD>(words[1], std::stod(words[2]) , std::stod(words[3]) , std::stoi(words[4]) ,
                                        std::stod(words[5]) , std::stod(words[6]) , std::stoi(words[7]), ScororPSD::YZ );
    }
  }
  else if(words[0]=="VolFlux")
  {
    std::cout << words[1] << " "
    <<  std::stod(words[2]) << " "
    <<  std::stod(words[3]) << " "
    <<  std::stoi(words[4]) << " "
    <<  std::stoi(words[5]) << " "
    <<  vol << std::endl;
    return std::make_shared<Prompt::ScororVolFlux>(words[1], std::stod(words[2]) ,
                std::stod(words[3]) , std::stoi(words[4]) ,  std::stoi(words[5]),
                vol );
  }
  else
    PROMPT_THROW2(BadInput, "Scoror type " << words[0] << " is not supported. ")
}
