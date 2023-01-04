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
#include "PTScorerRotatingObj.hh"
#include "PTScorerWlSpectrum.hh"

Prompt::ScorerFactory::ScorerFactory()
{}


std::shared_ptr<Prompt::Scorer> Prompt::ScorerFactory::createScorer(const std::string &cfgstr, double vol)
{
  std::cout << "Parsing scorer with config string: \n";
  std::cout << cfgstr << "\n";
  //fixme check number of input config

  auto &ps = Singleton<CfgParser>::getInstance();
  auto cfg = ps.parse(cfgstr);
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


      int parCount = 12;

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
      bool qtrue = true;
      std::string qtrueInStr = cfg.find("Qtrue");
      if(qtrueInStr.empty())
        parCount--;
      else
      {
        if(qtrueInStr=="yes")
        {
          qtrue = true;
        }
        else if(qtrueInStr=="no")
          qtrue = false;
        else {
          PROMPT_THROW2(BadInput, "The value for \"Qtrue\" should either be \"yes\" or \"no\"");
        }

      }

      int scatnum = -1;
      if(cfg.find("scatnum")=="") 
        parCount--;
      else
      {
        int scatnumInInt = ptstoi(cfg.find("scatnum"));
        if(scatnumInInt>=-1 )
        {
          scatnum = scatnumInInt;
        }
        else {
          PROMPT_THROW2(BadInput, "The value for \"scatnum\" should an integer greater than or equal to -1");
        }
      }

      Scorer::ScorerType ptstate = Scorer::ScorerType::ENTRY;
      std::string ptstateInStr = cfg.find("ptstate");
      if(ptstateInStr.empty())
        parCount--;
      else
      {
        if(ptstateInStr=="ENTRY")
        {
          ptstate = Scorer::ScorerType::ENTRY;
        }
        else if(ptstateInStr=="ABSORB")
        {
          ptstate = Scorer::ScorerType::ABSORB;
        }
        else {
          PROMPT_THROW(BadInput, "ptstate can only be ENTRY or ABSORB" );
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

      return std::make_shared<Prompt::ScorerNeutronSq>(name, samplePos, beamDir, moderator2SampleDist, minQ, maxQ, numBin, ptstate, qtrue, scatnum, linear);
    }

    else if(ScorDef == "PSD")
    {
      // PSD: position sensitive detector

      int parCount = 10;

      bool force = true;

      std::string name = cfg.find("name", force);
      double xmin = ptstod(cfg.find("xmin", force));
      double xmax = ptstod(cfg.find("xmax", force));
      int nxbins = ptstoi(cfg.find("numBins_x", force));
      double ymin = ptstod(cfg.find("ymin", force));
      double ymax = ptstod(cfg.find("ymax", force));
      int nybins = ptstoi(cfg.find("numBins_y", force));

      Scorer::ScorerType ptstate = Scorer::ScorerType::ENTRY;
      std::string ptstateInStr = cfg.find("ptstate");
      if(ptstateInStr.empty())
        parCount--;
      else
      {
        if(ptstateInStr=="ENTRY")
        {
          ptstate = Scorer::ScorerType::ENTRY;
        }
        else if(ptstateInStr=="ABSORB")
        {
          ptstate = Scorer::ScorerType::ABSORB;
        }
        else if(ptstateInStr=="SURFACE")
        {
          ptstate = Scorer::ScorerType::SURFACE;
        }
        else if(ptstateInStr=="PROPAGATE")
        {
          ptstate = Scorer::ScorerType::PROPAGATE;
        }
        else if(ptstateInStr=="EXIT")
        {
          ptstate = Scorer::ScorerType::EXIT;
        }
        else if(ptstateInStr=="ENTRY2EXIT")
        {
          ptstate = Scorer::ScorerType::ENTRY2EXIT;
        }
        else {
          PROMPT_THROW2(BadInput, "ptstate does not support" << " " << ptstateInStr);
        }
      }

      ScorerPSD::PSDType type = ScorerPSD::PSDType::XY;
      std::string typeInStr = cfg.find("type");
      if(typeInStr.empty())
        parCount--;
      else
      {
        if(typeInStr=="XY")
        {
          type = ScorerPSD::PSDType::XY;
        }
        else if(typeInStr=="XZ")
        {
          type = ScorerPSD::PSDType::XZ;
        }
        else if(typeInStr=="YZ")
        {
          type = ScorerPSD::PSDType::YZ;
        }
        else {
          PROMPT_THROW(BadInput, "Scorer type can only be XY, XZ or YZ" );
        }
      }

      if(parCount!=cfg.size())
      {
        PROMPT_THROW2(BadInput, "Scorer type PSD is missing or with extra config parameters" << cfg.size() << " " << parCount );
      }

      return std::make_shared<Prompt::ScorerPSD>(name, xmin, xmax, nxbins, ymin, ymax, nybins, ptstate,type);
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
    else if(ScorDef == "WlSpectrum")
    {
      // ESpectrum: energy spectrum
      // example cfg
      // ""Scorer=ESD; name=detector; Emin=0.0; Emax=0.0253; numbin=100""

      int parCount = 5;

      // The mandatory parameters
      bool force = true;
      std::string name = cfg.find("name", force);
      double minWl = ptstod(cfg.find("Wlmin", force));
      double maxWl = ptstod(cfg.find("Wlmax", force));
      int numBin = ptstoi(cfg.find("numbin", force));

      if(parCount!=cfg.size())
      {
        PROMPT_THROW2(BadInput, "Scorer type ESpectrum is missing or with extra config parameters" << cfg.size() << " " << parCount );
      }

      return std::make_shared<ScorerWlSpectrum>(name, minWl, maxWl, numBin);
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
      // ""Scorer=MultiScat; name=D2O; Numbermin=1; Numbermax=5; ptstate=PROPAGATE; linear=yes""
      // the default value for linear is yes

      int parCount = 6;

      // The mandatory parameters
      bool force = true;
      std::string name = cfg.find("name", force);
      double minNumber = ptstod(cfg.find("Numbermin", force));
      double maxNumber = ptstod(cfg.find("Numbermax", force));
      int numBin = maxNumber-minNumber+1;

      // the optional parameters
      Scorer::ScorerType ptstate = Scorer::ScorerType::PROPAGATE;
      std::string ptstateInStr = cfg.find("ptstate");
      if(ptstateInStr.empty())
        parCount--;
      else
      {
        if(ptstateInStr=="ENTRY")
        {
          ptstate = Scorer::ScorerType::ENTRY;
        }
        else if(ptstateInStr=="ABSORB")
        {
          ptstate = Scorer::ScorerType::ABSORB;
        }
        else if(ptstateInStr=="SURFACE")
        {
          ptstate = Scorer::ScorerType::SURFACE;
        }
        else if(ptstateInStr=="PROPAGATE")
        {
          ptstate = Scorer::ScorerType::PROPAGATE;
        }
        else if(ptstateInStr=="EXIT")
        {
          ptstate = Scorer::ScorerType::EXIT;
        }
        else if(ptstateInStr=="ENTRY2EXIT")
        {
          ptstate = Scorer::ScorerType::ENTRY2EXIT;
        }
        else {
          PROMPT_THROW2(BadInput, "ptstate does not support" << " " << ptstateInStr);
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
        PROMPT_THROW2(BadInput, "Scorer type MultiScat is missing or with extra config parameters" << cfg.size() << " " << parCount );
      }

      return std::make_shared<ScorerMultiScat>(name, minNumber-0.5, maxNumber+0.5, numBin, ptstate, linear);
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
    else if(ScorDef == "RotatingObj")
    {
      // "Scorer=RotatingObj;name=ro1;rotation_axis=0,1,0;point_on_axis=0,0,0;rot_fre=100;type=Entry or Proprogate or Exit"/

      int parCount = 5;
      bool force = true;

      std::string name = cfg.find("name", force);
      auto rotAxis = string2vec(cfg.find("rotation_axis", force));
      auto pointAxis = string2vec(cfg.find("point_on_axis", force));
      double rotFreq = ptstod(cfg.find("rot_fre", force));
      return std::make_shared<Prompt::ScorerRotatingObj>(name, rotAxis, pointAxis, rotFreq);
    }
    else
      PROMPT_THROW2(BadInput, "Scorer type " << ScorDef << " is not supported. ")
  }

}
