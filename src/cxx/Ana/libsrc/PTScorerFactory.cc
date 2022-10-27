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
#include "PTCfgParser.hh"

#include "PTScorerNeutronSq.hh"
#include "PTScorerPSD.hh"
#include "PTScorerESpectrum.hh"
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
      bool force = true;
      std::string name = cfg.find("name", force);
      auto samplePos = string2vec(cfg.find("sample_position", force));
      auto beamDir = string2vec(cfg.find("beam_direction", force));
      double moderator2SampleDist = ptstod(cfg.find("src_sample_dist", force));
      double minQ = ptstod(cfg.find("Qmin", force));
      double maxQ = ptstod(cfg.find("Qmax", force));
      int numBin = ptstoi(cfg.find("numbin", force));


      // the optional parameters
      Scorer::ScorerType type = Scorer::ScorerType::ENTRY;
      std::string typeInStr = cfg.find("type");
      if(typeInStr.empty())
        parCount--;
      else
      {
        if(typeInStr=="ENTRY")
        {
          type = Scorer::ScorerType::ENTRY;
        }
        else if(typeInStr=="ABSORB")
        {
          type = Scorer::ScorerType::ABSORB;
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
        else if(linearInStr=="no")
          linear = false;
        else {
          PROMPT_THROW2(BadInput, "The value for \"linear\" should either be \"yes\" or \"no\"");
        }

      }

      if(parCount!=cfg.size())
      {
        PROMPT_THROW2(BadInput, "Scorer type NeutronSq is missing or with extra config parameters" << cfg.size() << " " << parCount );
      }

      return std::make_shared<Prompt::ScorerNeutronSq>(name, samplePos, beamDir, moderator2SampleDist, minQ, maxQ, numBin, type, linear);
    }

    else if(ScorDef == "PSD")
    {
      // PSD: position sensitive detector

      int parCount = 9;

      bool force = true;

      std::string name = cfg.find("name", force);
      double xmin = ptstod(cfg.find("xmin", force));
      double xmax = ptstod(cfg.find("xmax", force));
      int nxbins = ptstoi(cfg.find("numBins_x", force));
      double ymin = ptstod(cfg.find("ymin", force));
      double ymax = ptstod(cfg.find("ymax", force));
      int nybins = ptstoi(cfg.find("numBins_y", force));

      ScorerPSD::ScorerType type = ScorerPSD::ScorerType::XY;
      std::string typeInStr = cfg.find("type");
      if(typeInStr.empty())
        parCount--;
      else
      {
        if(typeInStr=="XY")
        {
          type = ScorerPSD::ScorerType::XY;
        }
        else if(typeInStr=="XZ")
        {
          type = ScorerPSD::ScorerType::XZ;
        }
        else if(typeInStr=="YZ")
        {
          type = ScorerPSD::ScorerType::YZ;
        }
        else {
          PROMPT_THROW(BadInput, "Scorer type can only be XY, XZ or YZ" );
        }
      }

      if(parCount!=cfg.size())
      {
        PROMPT_THROW2(BadInput, "Scorer type PSD is missing or with extra config parameters" << cfg.size() << " " << parCount );
      }

      return std::make_shared<Prompt::ScorerPSD>(name, xmin, xmax, nxbins, ymin, ymax, nybins, type);
    }

    else if(ScorDef == "ESpectrum")
    {
      // ESpectrum: energy spectrum
      // example cfg
      // ""Scorer=ESD; name=detector; Emin=0.0; Emax=0.0253; numbin=100""

      int parCount = 5;

      // The mandatory parameters
      bool force = true;
      std::string name = cfg.find("name", force);
      double minE = ptstod(cfg.find("Emin", force));
      double maxE = ptstod(cfg.find("Emax", force));
      int numBin = ptstoi(cfg.find("numbin", force));

      if(parCount!=cfg.size())
      {
        PROMPT_THROW2(BadInput, "Scorer type ESpectrum is missing or with extra config parameters" << cfg.size() << " " << parCount );
      }

      return std::make_shared<ScorerESpectrum>(name, minE, maxE, numBin);
    }
    else if(ScorDef == "TOF")
    {
      // TOF: time-of-flight
      // example cfg
      // ""Scorer=TOF; name=detector; Tmin=0.0; Tmax=0.5; numbin=1000""

      int parCount = 5;

      // The mandatory parameters
      bool force = true;
      std::string name = cfg.find("name", force);
      double minT = ptstod(cfg.find("Tmin", force));
      double maxT = ptstod(cfg.find("Tmax", force));
      int numBin = ptstoi(cfg.find("numbin", force));

      if(parCount!=cfg.size())
      {
        PROMPT_THROW2(BadInput, "Scorer type TOF is missing or with extra config parameters" << cfg.size() << " " << parCount );
      }

      return std::make_shared<ScorerTOF>(name, minT, maxT, numBin);
    }
    else if(ScorDef == "MultiScat")
    {
      // MultiScat: multiple scattering
      // example cfg
      // ""Scorer=MultiScat; name=D2O; Numbermin=1; Numbermax=5; linear=yes""
      // the default value for linear is yes

      int parCount = 5;

      // The mandatory parameters
      bool force = true;
      std::string name = cfg.find("name", force);
      double minNumber = ptstod(cfg.find("Numbermin", force));
      double maxNumber = ptstod(cfg.find("Numbermax", force));
      int numBin = maxNumber-minNumber+1;

      // the optional parameters
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
        else if(linearInStr=="no")
          linear = false;
        else {
          PROMPT_THROW2(BadInput, "The value for \"linear\" should either be \"yes\" or \"no\"");
        }
      }

      if(parCount!=cfg.size())
      {
        PROMPT_THROW2(BadInput, "Scorer type MultiScat is missing or with extra config parameters" << cfg.size() << " " << parCount );
      }

      return std::make_shared<ScorerMultiScat>(name, minNumber-0.5, maxNumber+0.5, numBin, linear);
    }
    else if(ScorDef == "VolFlux")
    {
      // VolFlux: volume flux
      
      int parCount = 6;

      bool force = true;
      std::string name = cfg.find("name", force);
      double xmin = ptstod(cfg.find("xmin", force));
      double xmax = ptstod(cfg.find("xmax", force));
      int nxbins = ptstoi(cfg.find("numBins_x", force));
      
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
        else if(linearInStr=="no")
          linear = false;
        else {
          PROMPT_THROW2(BadInput, "The value for \"linear\" should either be \"yes\" or \"no\"");
        }

      }

      if(parCount!=cfg.size())
      {
        PROMPT_THROW2(BadInput, "Scorer type VolFlux is missing or with extra config parameters" << cfg.size() << " " << parCount );
      }

      return std::make_shared<Prompt::ScorerVolFlux>(name, xmin, xmax, nxbins, linear, vol);

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
  //     return std::make_shared<Prompt::ScorerNeutronSq>(words[1], samplePos, beamDir, moderator2SampleDist, minQ, maxQ, numBin, Prompt::Scorer::ScorerType::ABSORB);
  //   else if(words[8]=="ENTRY")
  //     return std::make_shared<Prompt::ScorerNeutronSq>(words[1], samplePos, beamDir, moderator2SampleDist, minQ, maxQ, numBin, Prompt::Scorer::ScorerType::ENTRY);
  //   else
  //   {
  //     PROMPT_THROW2(BadInput, words[8] << " type is not supported by ScorerNeutronSq");
  //     return std::make_shared<Prompt::ScorerNeutronSq>(words[1], samplePos, beamDir, moderator2SampleDist, minQ, maxQ, numBin, Prompt::Scorer::ScorerType::ENTRY);
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
