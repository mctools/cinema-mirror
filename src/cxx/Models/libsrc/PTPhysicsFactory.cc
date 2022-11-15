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
#include "NCrystal/NCrystal.hh"
#include "PTNCrystalScat.hh"
#include "PTNCrystalAbs.hh"


bool Prompt::PhysicsFactory::pureNCrystalCfg(const std::string &cfgstr)
{
  try
  {
    auto &ps = Singleton<CfgParser>::getInstance();
    CfgParser::ScorerCfg cfg = ps.parse(cfgstr);
  }
  catch (Prompt::Error::BadInput& e)
  {
    return true;
  }
  return false;

}

double Prompt::PhysicsFactory::calNumDensity(const std::string &cfgstr)
{
  std::string nccfgstr = cfgstr;
  if(!pureNCrystalCfg(cfgstr))
  {
    CfgParser::ScorerCfg cfg = Singleton<CfgParser>::getInstance().parse(cfgstr);
    if(!cfg.getStringIfExist("nccfg", nccfgstr))
      PROMPT_THROW(BadInput, "NCrystal cfg string is not found");
  }

  // fixme: encounter what():  Assertion failure: isSingleIsotope() from NCrystal::AtomData::A when 
  // the material is LiquidHeavyWaterD2O_T293.6K.ncmat
  // showComposition(nccfgstr); 

  NCrystal::MatCfg matcfg(nccfgstr);
  auto info = NCrystal::createInfo(matcfg);
  if(info->hasNumberDensity())
    return info->getNumberDensity().get() / Unit::Aa3;
  else
  {
    PROMPT_THROW2(CalcError, "material has no number density " << nccfgstr);
    return 0.;
  }
}

void Prompt::PhysicsFactory::showComposition(const std::string &nccfgstr)
{
  NCrystal::MatCfg matcfg(nccfgstr);
  auto info = NCrystal::createInfo(matcfg);
  const NCrystal::Info::Composition & comp = info->getComposition();


  for(const NCrystal::Info::CompositionEntry &v : comp)
  {
    double frac = v.fraction;
    const auto& atom = v.atom.data();
    std::cout << atom.elementName() << ": A " << atom.A() << ", Z " << atom.Z() << ", fraction " << frac << std::endl;
  }

}


std::unique_ptr<Prompt::CompoundModel> Prompt::PhysicsFactory::createBulkPhysics(const std::string &cfgstr)
{
  std::cout << "Parsing config string for a CompoundModel: \n";
  CfgParser::ScorerCfg cfg = Singleton<CfgParser>::getInstance().parse(cfgstr);
  cfg.print();

  std::unique_ptr<Prompt::CompoundModel> compmod;
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
      int parCount = 4; 
      std::string nccfg = cfg.find("nccfg", true);

      double scatter_bias = 0.;
      if(!cfg.getDoubleIfExist("scatter_bias", scatter_bias))
        parCount--;

      double abs_bias = 0.;
      if(!cfg.getDoubleIfExist("abs_bias", abs_bias))
        parCount--;

      if(!scatter_bias && !abs_bias)  
      {
        PROMPT_THROW2(BadInput, "At lease one of the \"scatter_bias\" and \"abs_bias\" key shoule be set to a non-zero positive value" );
      }

      auto compm = std::make_unique<CompoundModel> (2112);
      if(scatter_bias)
      {
        compm->addPhysicsModel(std::make_shared<NCrystalScat>(nccfg, scatter_bias));
      }
      if(abs_bias)
      {
        compm->addPhysicsModel(std::make_shared<NCrystalAbs>(nccfg, abs_bias));
      }

      return compm;   
    }
  }
}


std::shared_ptr<Prompt::BoundaryPhysics> Prompt::PhysicsFactory::createBoundaryPhysics(const std::string &cfgstr)
{
  std::cout << "Parsing config string for a physics model: \n";

  auto &ps = Singleton<CfgParser>::getInstance();
  auto cfg = ps.parse(cfgstr);
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
      double m = 0;
      if(!cfg.getDoubleIfExist("m", m))
        parCount--;

      double threshold =  1e-3;
      if(!cfg.getDoubleIfExist("threshold", threshold))
        parCount--;

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
