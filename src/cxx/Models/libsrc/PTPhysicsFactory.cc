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

#include "PTPhysicsFactory.hh"
#include "PTUtils.hh"
#include "PTCfgParser.hh"

#include "PTMirrorPhysics.hh"

std::shared_ptr<Prompt::CompoundModel> Prompt::PhysicsFactory::createBulkPhysics(const std::string &cfgstr)
{
  std::cout << "Parsing config string for a CompoundModel: \n";
  auto &ps = Singleton<CfgParser>::getInstance();
  auto cfg = ps.getScorerCfg(cfgstr);
  std::cout << "Parsed cfg: \n";
  cfg.print();

  std::string physDef = cfg.find("physics");

  if(physDef.empty())
  {
    PROMPT_THROW2(BadInput, "Config string " << cfgstr << " does not define the scorer by the key \"physics\" ")
  }
  else
  {
    // example cfg
    // physics=ncrystal; nccfg="LiquidHeavyWaterD2O_T293.6K.ncmat;density=1.0gcm3";scatter_bias=1.0;abs_bias=1.0;
    if(physDef == "ncrystal")
    {
      std::string nccfg = cfg.find("nccfg", true);
    }
  }
}


std::shared_ptr<Prompt::BoundaryPhysics> Prompt::PhysicsFactory::createBoundaryPhysics(const std::string &cfgstr)
{
  std::cout << "Parsing config string for a physics model: \n";

  auto &ps = Singleton<CfgParser>::getInstance();
  auto cfg = ps.getScorerCfg(cfgstr);
  std::cout << "Parsed cfg: \n";
  cfg.print();

  std::string physDef = cfg.find("physics");

  if(physDef.empty())
  {
    PROMPT_THROW2(BadInput, "Config string " << cfgstr << " does not define the scorer by the key \"physics\" ")
  }
  else
  {
    std::shared_ptr<BoundaryPhysics> phy;

    if(physDef == "MirrorPhyiscs")
    {

      // example cfg
      // ""physics=MirrorPhyiscs; m=1.0; threshold=1e-5""

      // where the m value is 1 by default//
      // the default threshold for the Russian roulette biasing method to be activated

      int parCount = 3;

      // optional parameters
      double m = 1.0;
      std::string mInStr = cfg.find("m");
      if(mInStr.empty())
        parCount--;
      else
      {
        m = ptstod(mInStr);
      }

      double threshold = 1e-3;
      std::string thresholdInStr = cfg.find("threshold");
      if(thresholdInStr.empty())
        parCount--;
      else
      {
        threshold = ptstod(thresholdInStr);
      }

      if(parCount!=cfg.size())
      {
        PROMPT_THROW2(BadInput, "Cfgstr for a mirror physics is missing or with extra config parameters" << cfg.size() << " " << parCount );
      }

      phy = std::make_shared<MirrorPhyiscs>(m, threshold);
    }

    if(phy)
    {
      return phy;
    }
    else
      PROMPT_THROW2(BadInput, "Physics type " << physDef << " is not supported. ")
  }

}
