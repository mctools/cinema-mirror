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
#include "PTMCPLGun.hh"
Prompt::GunFactory::GunFactory()
{}


std::shared_ptr<Prompt::PrimaryGun> Prompt::GunFactory::createGun(const std::string &cfgstr)
{
  std::cout << "Parsing config string for the particle gun: \n";
  std::cout << cfgstr << "\n";
  //fixme check number of input config

  auto &ps = Singleton<CfgParser>::getInstance();
  auto cfg = ps.parse(cfgstr);
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
      // example cfgstr:
      // gun=SimpleThermalGun;energy=0.0253;position=0,0,-500;direction=0,0,1

      // energy and direction are optional with default value 0 and 0,0,1 separately

      int parCount = 4;

      // The mandatory parameters
      bool force = true;
      Vector position = string2vec(cfg.find("position", force));

      // the optional parameters
      double energy = 0;
      if(!cfg.getDoubleIfExist("energy", energy))
        parCount--;

      Vector direction = Vector{0.,0.,1.};
      if(!cfg.getVectorIfExist("direction", direction))
        parCount--;

      if(parCount!=cfg.size())
      {
        PROMPT_THROW2(BadInput, "SimpleThermalGun is missing or with extra config parameters" << cfg.size() << " " << parCount );
      }

      return std::make_shared<SimpleThermalGun>(Neutron(), energy, position, direction);
    }
    else if(gunDef == "IsotropicGun")
    {
      // example cfgstr:
      // gun=IsotropicGun;energy=0.0253;position=0,0,-10

      // energy is optional with default value 0

      int parCount = 3;

      // The mandatory parameters
      bool force = true;
      Vector position = string2vec(cfg.find("position", force));

      // the optional parameters
      double energy = 0;
      std::string energyInStr = cfg.find("energy");
      if(energyInStr.empty())
        parCount--;
      else
      {
        energy = ptstod(energyInStr);
      }

      Vector direction = Vector{0.,0.,1.};

      if(parCount!=cfg.size())
      {
        PROMPT_THROW2(BadInput, "IsotropicGun is missing or with extra config parameters" << cfg.size() << " " << parCount );
      }

      return std::make_shared<IsotropicGun>(Neutron(), energy, position, direction);
    }
    else if(gunDef == "UniModeratorGun")
    {
      // example cfgstr:
      // gun=UniModeratorGun;mean_wavelength=3.39;range_wavelength=0.3;moderator_width_x=100;moderator_height_y=50;moderator_positon_z=-400;
      // slit_width_x=5;slit_height_y=10;slit_position_z=1


      int parCount = 9;

      // The mandatory parameters
      bool force = true;
      double mean_wavelength = ptstod(cfg.find("mean_wavelength",force));
      double range_wavelength = ptstod(cfg.find("range_wavelength",force));
      double moderator_width_x = ptstod(cfg.find("moderator_width_x", force));
      double moderator_height_y = ptstod(cfg.find("moderator_height_y", force));
      double moderator_positon_z = ptstod(cfg.find("moderator_positon_z", force));
      double slit_width_x = ptstod(cfg.find("slit_width_x", force));
      double slit_height_y = ptstod(cfg.find("slit_height_y", force));
      double slit_position_z = ptstod(cfg.find("slit_position_z", force));

      if(parCount!=cfg.size())
      {
        PROMPT_THROW2(BadInput, "UniModeratorGun is missing or with extra config parameters" << cfg.size() << " " << parCount );
      }

      return std::make_shared<UniModeratorGun>(Neutron(), mean_wavelength, range_wavelength,
            std::array<double, 6> {moderator_width_x, moderator_height_y, moderator_positon_z,
                                 slit_width_x, slit_height_y, slit_position_z});
    }
    else if(gunDef == "MPIGun")
    {
      // example cfgstr:
      // gun=MPIGun;moderator_width_x=100;moderator_height_y=50;moderator_positon_z=-400;
      // slit_width_x=5;slit_height_y=10;slit_position_z=1

      int parCount = 7;

      // The mandatory parameters
      bool force = true;
      double moderator_width_x = ptstod(cfg.find("moderator_width_x", force));
      double moderator_height_y = ptstod(cfg.find("moderator_height_y", force));
      double moderator_positon_z = ptstod(cfg.find("moderator_positon_z", force));
      double slit_width_x = ptstod(cfg.find("slit_width_x", force));
      double slit_height_y = ptstod(cfg.find("slit_height_y", force));
      double slit_position_z = ptstod(cfg.find("slit_position_z", force));

      if(parCount!=cfg.size())
      {
        PROMPT_THROW2(BadInput, "MPIGun is missing or with extra config parameters" << cfg.size() << " " << parCount );
      }

      return std::make_shared<MPIGun>(Neutron(),
            std::array<double, 6> {moderator_width_x, moderator_height_y, moderator_positon_z,
                                 slit_width_x, slit_height_y, slit_position_z});
    }
    else if(gunDef == "MCPLGun")
    {
      std::string fn = cfg.find("mcplfile", true);
      return std::make_shared<MCPLGun>(Neutron(), fn);
    }
    else
      PROMPT_THROW2(BadInput, "Gun " << gunDef << " is not supported. ")
  }
  assert(false);
  //a return statement here to stop compiler warning, it should never reach here
  return std::make_shared<IsotropicGun>(Neutron(), 1, Vector{0,0,0}, Vector{1,0,0});

}
