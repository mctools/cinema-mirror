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

#include "PTGunFactory.hh"
#include "PTUtils.hh"
#include "PTCfgParser.hh"

#include "PTNeutron.hh"

#include "PTMaxwellianGun.hh"
#include "PTSimpleThermalGun.hh"
#include "PTIsotropicGun.hh"
#include "PTUniModeratorGun.hh"
#include "PTMPIGun.hh"
Prompt::GunFactory::GunFactory()
{}


std::shared_ptr<Prompt::PrimaryGun> Prompt::GunFactory::createGun(const std::string &cfgstr)
{
  std::cout << "Parsing config string for the particle gun: \n";
  std::cout << cfgstr << "\n";
  //fixme check number of input config

  auto &ps = Singleton<CfgParser>::getInstance();
  auto cfg = ps.getScorerCfg(cfgstr);
  std::cout << "Parsed cfg: \n";
  cfg.print();

  std::string gunDef = cfg.find("gun");

  if(gunDef.empty())
  {
    PROMPT_THROW2(BadInput, "Scorer config string " << cfgstr << " does not define the scorer by the key \"Scorer\" ")
  }
  else
  {

    if(gunDef == "MaxwellianGun")
    {
      // example cfgstr:
      // gun=MaxwellianGun;moderator_width_x=100;moderator_height_y=50;moderator_positon_z=-10;
      // slit_width_x=5;slit_height_y=10;slit_position_z=10;temperature=293.15;

      // temperature is optional with default value 293.15kelvin

      int parCount = 8;

      // The mandatory parameters
      bool force = true;
      double moderator_width_x = ptstod(cfg.find("moderator_width_x", force));
      double moderator_height_y = ptstod(cfg.find("moderator_height_y", force));
      double moderator_positon_z = ptstod(cfg.find("moderator_positon_z", force));
      double slit_width_x = ptstod(cfg.find("slit_width_x", force));
      double slit_height_y = ptstod(cfg.find("slit_height_y", force));
      double slit_position_z = ptstod(cfg.find("slit_position_z", force));

      // the optional parameters
      double temp = 293.15;
      std::string temperatureInStr = cfg.find("temperature");
      if(temperatureInStr.empty())
        parCount--;
      else
      {
        temp=ptstod(temperatureInStr);
      }

      if(parCount!=cfg.size())
      {
        PROMPT_THROW2(BadInput, "MaxwellianGun is missing or with extra config parameters" << cfg.size() << " " << parCount );
      }

     return std::make_shared<MaxwellianGun>(Neutron(), temp,
          std::array<double, 6> {moderator_width_x, moderator_height_y, moderator_positon_z,
                                 slit_width_x, slit_height_y, slit_position_z});
    }
    else if(gunDef == "SimpleThermalGun")
    {
      PROMPT_THROW(LogicError, "not yet impletmented ")
    }
    else if(gunDef == "IsotropicGun")
    {
      PROMPT_THROW(LogicError, "not yet impletmented ")
    }
    else if(gunDef == "UniModeratorGun")
    {
      PROMPT_THROW(LogicError, "not yet impletmented ")
    }
    else if(gunDef == "TMPIGun")
    {
      PROMPT_THROW(LogicError, "not yet impletmented ")
    }
    else
      PROMPT_THROW2(BadInput, "Gun " << gunDef << " is not supported. ")
  }


        // auto words = split(info.GetValue(), ';');
        // if(words[0]=="MaxwellianGun")
        // {
        //   double temp = ptstod(words[2]);
        //   auto positions = split(words[3], ',');
        //
        //   m_gun = std::make_shared<MaxwellianGun>(Neutron(), temp,
        //     std::array<double, 6> {ptstod(positions[0]), ptstod(positions[1]), ptstod(positions[2]),
        //                            ptstod(positions[3]), ptstod(positions[4]), ptstod(positions[5])});
        // }
        // else if(words[0]=="MPIGun")
        // {
        //   auto positions = split(words[2], ',');
        //   m_gun = std::make_shared<MPIGun>(Neutron(),
        //     std::array<double, 6> {ptstod(positions[0]), ptstod(positions[1]), ptstod(positions[2]),
        //                            ptstod(positions[3]), ptstod(positions[4]), ptstod(positions[5])});
        // }
        // else if(words[0]=="UniModeratorGun")
        // {
        //   double wl0 = ptstod(words[2]);
        //   double wl_dlt = ptstod(words[3]);
        //   auto positions = split(words[4], ',');
        //
        //   m_gun = std::make_shared<UniModeratorGun>(Neutron(), wl0, wl_dlt,
        //     std::array<double, 6> {ptstod(positions[0]), ptstod(positions[1]), ptstod(positions[2]),
        //                            ptstod(positions[3]), ptstod(positions[4]), ptstod(positions[5])});
        // }
        // else if(words[0]=="SimpleThermalGun")
        // {
        //   double ekin = ptstod(words[2]);
        //   m_gun = std::make_shared<SimpleThermalGun>(Neutron(), ekin, string2vec(words[3]), string2vec(words[4]));
        // }
        // else if(words[0]=="IsotropicGun")
        // {
        //   double ekin = ptstod(words[2]);
        //   m_gun = std::make_shared<IsotropicGun>(Neutron(), ekin, string2vec(words[3]), string2vec(words[4]));
        // }
        // else
        //   PROMPT_THROW2(BadInput, "No such gun");
}
