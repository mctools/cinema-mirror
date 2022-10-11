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

#include "PTScorerFactory.hh"
#include "PTUtils.hh"
#include "PTScorerNeutronSq.hh"
#include "PTScorerPSD.hh"
#include "PTScorerESD.hh"
#include "PTScorerTOF.hh"
#include "PTScorerVolFlux.hh"
#include "PTScorerMultiScat.hh"

Prompt::ScorerFactory::ScorerFactory()
{}


std::shared_ptr<Prompt::Scorer> Prompt::ScorerFactory::createScorer(const std::string &cfgstr, double vol)
{
  std::cout << "Parsing scorer with config string: \n";
  std::cout << cfgstr << "\n";
  //fixme check number of input config

  auto &ps = Singleton<CfgParser>::getInstance();
  auto cfg = ps.getScorerCfg(cfgstr);
  std::cout << "Parsed cfg: \n";
  cfg.print();

  std::string ScorDef = cfg.find("Scorer");

  if(ScorDef.empty())
  {
    PROMPT_THROW2(BadInput, "Scorer config string " << cfgstr << " does not define the scorer by the key \"Scorer\" ")
  }
  else
  {

    if(ScorDef == "NeutronSq")
    {
      // example cfg
      // ""Scorer=NeutronSq; name=SofQ;sample_position=0,0,1;beam_direction=0,0,1;src_sample_dist=-100;
      // type=ENTRY; linear=yes; Qmin=0.5; Qmax = 50; numbin=1000""

      // where type can be ENTRY(default) or ABSORB, the default value for linear is yes


      int parCount = 10;

      // The mandatory parameters
      if (!cfg.contains("name"))
        PROMPT_THROW(BadInput, "Scorer cfg should define the key \"name\"" );
      std::string name = cfg.find("name");

      if (!cfg.contains("sample_position"))
        PROMPT_THROW(BadInput, "Scorer cfg should define the key \"sample_position\"" );
      auto samplePos = string2vec(cfg.find("sample_position"));

      if (!cfg.contains("beam_direction"))
        PROMPT_THROW(BadInput, "Scorer cfg should define the key \"beam_direction\"" );
      auto beamDir = string2vec(cfg.find("beam_direction"));

      if (!cfg.contains("src_sample_dist"))
        PROMPT_THROW(BadInput, "Scorer cfg should define the key \"src_sample_dist\"" );
      double moderator2SampleDist = ptstod(cfg.find("src_sample_dist"));

      if (!cfg.contains("Qmin"))
        PROMPT_THROW(BadInput, "Scorer cfg should define the key \"Qmin\"" );
      double minQ = ptstod(cfg.find("Qmin"));

      if (!cfg.contains("Qmax"))
        PROMPT_THROW(BadInput, "Scorer cfg should define the key \"Qmax\"" );
      double maxQ = ptstod(cfg.find("Qmax"));

      if (!cfg.contains("numbin"))
        PROMPT_THROW(BadInput, "Scorer cfg should define the key \"numbin\"" );
      int numBin = ptstoi(cfg.find("numbin"));


      // the optional parameters
      Scorer::ScorerType type = Scorer::ENTRY;
      std::string typeInStr = cfg.find("type");
      if(typeInStr.empty())
        parCount--;
      else
      {
        if(typeInStr=="ENTRY")
        {
          type = Scorer::ENTRY;
        }
        else if(typeInStr=="ABSORB")
        {
          type = Scorer::ABSORB;
        }
        else {
          PROMPT_THROW(BadInput, "Scorer type can only be ENTRY or ABSORB" );
        }
      }

      bool linear = true;
      std::string linearInStr = cfg.find("linear");
      if(linearInStr.empty())
        parCount--;
      else
      {
        if(linearInStr=="yes")
        {
          linear = true;
        }
        else
          linear = false;
      }

      if(parCount!=cfg.size())
      {
        PROMPT_THROW2(BadInput, "Scorer type NeutronSq is missing or with extra config parameters" << cfg.size() << " " << parCount );
      }

      return std::make_shared<Prompt::ScorerNeutronSq>(name, samplePos, beamDir, moderator2SampleDist, minQ, maxQ, numBin, Prompt::Scorer::ABSORB);
    }

    else if(ScorDef == "PSD")
    {

    }
    else if(ScorDef == "ESD")
    {

    }
    else if(ScorDef == "TOF")
    {

    }
    else if(ScorDef == "MultiScat")
    {

    }
    else if(ScorDef == "VolFlux")
    {

    }
    else
      PROMPT_THROW2(BadInput, "Scorer type " << ScorDef << " is not supported. ")
  }

  // if(words[0]=="NeutronSq")
  // {
  //   //type
  //   auto samplePos = string2vec(words[2]);
  //   auto beamDir = string2vec(words[3]);
  //   double moderator2SampleDist = ptstod(words[4]);
  //   double minQ = ptstod(words[5]);
  //   double maxQ = ptstod(words[6]);
  //   int numBin = std::stoi(words[7]);
  //   if(words[8]=="ABSORB")
  //     return std::make_shared<Prompt::ScorerNeutronSq>(words[1], samplePos, beamDir, moderator2SampleDist, minQ, maxQ, numBin, Prompt::Scorer::ABSORB);
  //   else if(words[8]=="ENTRY")
  //     return std::make_shared<Prompt::ScorerNeutronSq>(words[1], samplePos, beamDir, moderator2SampleDist, minQ, maxQ, numBin, Prompt::Scorer::ENTRY);
  //   else
  //   {
  //     PROMPT_THROW2(BadInput, words[8] << " type is not supported by ScorerNeutronSq");
  //     return std::make_shared<Prompt::ScorerNeutronSq>(words[1], samplePos, beamDir, moderator2SampleDist, minQ, maxQ, numBin, Prompt::Scorer::ENTRY);
  //   }
  // }
  // else if(words[0]=="PSD")
  // {
  //   if(words[8]=="XY")
  //       return std::make_shared<ScorerPSD>(words[1], ptstod(words[2]) , ptstod(words[3]) , std::stoi(words[4]) ,
  //                                         ptstod(words[5]) , ptstod(words[6]) , std::stoi(words[7]), ScorerPSD::XY );
  //   else if(words[8]=="XZ")
  //       return std::make_shared<ScorerPSD>(words[1], ptstod(words[2]) , ptstod(words[3]) , std::stoi(words[4]) ,
  //                                         ptstod(words[5]) , ptstod(words[6]) , std::stoi(words[7]), ScorerPSD::XZ );
  //   else if(words[8]=="YZ")
  //       return std::make_shared<ScorerPSD>(words[1], ptstod(words[2]) , ptstod(words[3]) , std::stoi(words[4]) ,
  //                                         ptstod(words[5]) , ptstod(words[6]) , std::stoi(words[7]), ScorerPSD::YZ );
  //   else
  //   {
  //     PROMPT_THROW2(BadInput, words[8] << " type is not supported by ScorerPSD");
  //     return std::make_shared<ScorerPSD>(words[1], ptstod(words[2]) , ptstod(words[3]) , std::stoi(words[4]) ,
  //                                       ptstod(words[5]) , ptstod(words[6]) , std::stoi(words[7]), ScorerPSD::YZ );
  //   }
  // }
  // else if(words[0]=="ESD")
  // {
  //   return std::make_shared<ScorerESD>(words[1], ptstod(words[2]) , ptstod(words[3]) , std::stoi(words[4]) );
  // }
  // else if(words[0]=="TOF")
  // {
  //   return std::make_shared<ScorerTOF>(words[1], ptstod(words[2]) , ptstod(words[3]) , std::stoi(words[4]) );
  // }
  // else if(words[0]=="MultiScat")
  // {
  //   return std::make_shared<ScorerMultiScat>(words[1], ptstod(words[2]) , ptstod(words[3]) , std::stoi(words[4]) );
  // }
  // else if(words[0]=="VolFlux")
  // {
  //   return std::make_shared<Prompt::ScorerVolFlux>(words[1], ptstod(words[2]) ,
  //               ptstod(words[3]) , std::stoi(words[4]) ,  std::stoi(words[5]),
  //               vol );
  // }
  // else
  //   PROMPT_THROW2(BadInput, "Scorer type " << words[0] << " is not supported. ")
}
