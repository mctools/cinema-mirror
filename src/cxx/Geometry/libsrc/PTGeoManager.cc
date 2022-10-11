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

#include "PTGeoManager.hh"

#include <VecGeom/base/Config.h>
#include <VecGeom/management/GeoManager.h>
#include <VecGeom/gdml/Middleware.h>
#include <VecGeom/gdml/Frontend.h>
#include <VecGeom/volumes/PlacedVolume.h>
#include "VecGeom/management/GeoManager.h"
#include "VecGeom/navigation/BVHNavigator.h"
#include "VecGeom/navigation/NewSimpleNavigator.h"
#include "VecGeom/navigation/SimpleABBoxNavigator.h"
#include "VecGeom/navigation/SimpleABBoxLevelLocator.h"
#include "VecGeom/navigation/BVHLevelLocator.h"

#include "PTScorerFactory.hh"

#include "PTUtils.hh"
#include "PTMaxwellianGun.hh"
#include "PTSimpleThermalGun.hh"
#include "PTIsotropicGun.hh"
#include "PTUniModeratorGun.hh"
#include "PTNeutron.hh"
#include "PTMPIGun.hh"

Prompt::GeoManager::GeoManager()
:m_gun(nullptr)
{
}

Prompt::GeoManager::~GeoManager()
{
  std::cout << "Simulation completed!\n";
  std::cout << "Simulation created " << numMaterialPhysics() << " material physics\n";
  std::cout << "There are " << numScorer() << " scorers in total\n";
}

std::shared_ptr<Prompt::MaterialPhysics> Prompt::GeoManager::getMaterialPhysics(const std::string &name)
{
  auto it = m_globelPhysics.find(name);
  if(it!=m_globelPhysics.end())
  {
    return it->second;
  }
  else
    return nullptr;
}

std::string Prompt::GeoManager::getLogicalVolumeScorerName(unsigned logid)
{
  std::string names;
  auto it = m_logVolID2physcorer.find(logid);
  if(it!=m_logVolID2physcorer.end())
  {
    for(const auto &sc : it->second->scorers)
    {
      names += sc->getName() + " ";
    }
  }
  return names;
}

const std::string &Prompt::GeoManager::getLogicalVolumeMaterialName(unsigned logid)
{
  auto it = m_logVolID2Mateiral.find(logid);
  assert(it!=m_logVolID2Mateiral.end());
  return it->second;
}

std::shared_ptr<Prompt::Scorer> Prompt::GeoManager::getScorer(const std::string &name)
{
  auto it = m_globelScorers.find(name);
  if(it!= m_globelScorers.end())
  {
    return it->second;
  }
  else
    return nullptr;
}


void Prompt::GeoManager::loadFile(const std::string &gdml_file)
{
  vgdml::Parser p;
  const auto loadedMiddleware = p.Load(gdml_file.c_str(), false, 1);

  //accelaration
  vecgeom::BVHManager::Init();
  for (auto &lvol : vecgeom::GeoManager::Instance().GetLogicalVolumesMap()) {
    auto ndaughters = lvol.second->GetDaughtersp()->size();

    if (ndaughters <= 2)
      lvol.second->SetNavigator(vecgeom::NewSimpleNavigator<>::Instance());
    else
      lvol.second->SetNavigator(vecgeom::BVHNavigator<>::Instance());

    if (lvol.second->ContainsAssembly())
      lvol.second->SetLevelLocator(vecgeom::SimpleAssemblyAwareABBoxLevelLocator::GetInstance());
    else
      lvol.second->SetLevelLocator(vecgeom::BVHLevelLocator::GetInstance());
  }


  if (!loadedMiddleware) PROMPT_THROW(DataLoadError, "failed to load the gdml file ");

  const auto &aMiddleware = *loadedMiddleware;
  auto volumeMatMap   = aMiddleware.GetVolumeMatMap();

  // Get User info, which includes primary generator definition
  auto uinfo = aMiddleware.GetUserInfo();
  for(const auto& info : uinfo)
  {
    std::cout << info.GetType() << std::endl;
    std::cout << info.GetValue() << std::endl;

    if(info.GetType()=="PrimaryGun")
    {
      auto words = split(info.GetValue(), ';');
      if(words[0]=="MaxwellianGun")
      {
        double temp = std::stod(words[2]);
        auto positions = split(words[3], ',');

        m_gun = std::make_shared<MaxwellianGun>(Neutron(), temp,
          std::array<double, 6> {std::stod(positions[0]), std::stod(positions[1]), std::stod(positions[2]),
                                 std::stod(positions[3]), std::stod(positions[4]), std::stod(positions[5])});
      }
      else if(words[0]=="MPIGun")
      {
        auto positions = split(words[2], ',');
        m_gun = std::make_shared<MPIGun>(Neutron(),
          std::array<double, 6> {std::stod(positions[0]), std::stod(positions[1]), std::stod(positions[2]),
                                 std::stod(positions[3]), std::stod(positions[4]), std::stod(positions[5])});
      }
      else if(words[0]=="UniModeratorGun")
      {
        double wl0 = std::stod(words[2]);
        double wl_dlt = std::stod(words[3]);
        auto positions = split(words[4], ',');

        m_gun = std::make_shared<UniModeratorGun>(Neutron(), wl0, wl_dlt,
          std::array<double, 6> {std::stod(positions[0]), std::stod(positions[1]), std::stod(positions[2]),
                                 std::stod(positions[3]), std::stod(positions[4]), std::stod(positions[5])});
      }
      else if(words[0]=="SimpleThermalGun")
      {
        double ekin = std::stod(words[2]);
        m_gun = std::make_shared<SimpleThermalGun>(Neutron(), ekin, string2vec(words[3]), string2vec(words[4]));
      }
      else if(words[0]=="IsotropicGun")
      {
        double ekin = std::stod(words[2]);
        m_gun = std::make_shared<IsotropicGun>(Neutron(), ekin, string2vec(words[3]), string2vec(words[4]));
      }
      else
        PROMPT_THROW2(BadInput, "No such gun");
    }
  }


  // Get the volume auxiliary info
  const std::map<int, std::vector<vgdml::Auxiliary>>& volAuxInfo = aMiddleware.GetVolumeAuxiliaryInfo();
  std::cout << "Geometry contains "
            << volAuxInfo.size() << " entries of volum auxiliary info\n";

  auto &scorerFactory = Singleton<ScorerFactory>::getInstance();
  auto &geoManager = vecgeom::GeoManager::Instance();

  //geoManager.GetLogicalVolumesMap() returens std::map<unsigned int, LogicalVolume *>
  for (const auto &item : geoManager.GetLogicalVolumesMap())
  {
    auto &volume   = *item.second;
    const size_t volID = volume.id();

    std::shared_ptr<VolumePhysicsScorer> vps(nullptr);
    if(m_logVolID2physcorer.find(volID)==m_logVolID2physcorer.end())
    {
      m_logVolID2physcorer.insert(std::make_pair(volID,  std::make_shared<VolumePhysicsScorer>()));
      vps = m_logVolID2physcorer[volID];
    }
    else
    {
      PROMPT_THROW2(CalcError, "volume ID " << volID << " appear more than once")
    }

    // 1. filter out material-empty volume
    auto mat_iter = volumeMatMap.find(volID);
    if(mat_iter==volumeMatMap.end()) //union creates empty virtual volume
    {
      m_logVolID2physcorer.erase(volID);
      // PROMPT_THROW(CalcError, "empty volume ")
      continue;
    }

    // 2. setup scorers
    if(volAuxInfo.size())
    {
      auto volAuxInfo_iter = volAuxInfo.find(volID);
      if(volAuxInfo_iter != volAuxInfo.end()) //it volume contains an AuxInfo info
      {
        std::cout << volume.GetName()<< ", volID " << volID << " contains volAuxInfo\n";
        const std::vector<vgdml::Auxiliary> &volAuxInfoVec = (*volAuxInfo_iter).second;
        auto volAuxInfoSize = volAuxInfoVec.size();
        std::cout << "volAuxInfoSize " << volume.GetName()  << " "
              << volAuxInfoSize << std::endl;

        for(const auto& info : volAuxInfoVec)
        {
          if (info.GetType() == "Scorer")
          {
            std::shared_ptr<Prompt::Scorer> scor = getScorer(info.GetValue());

            if(scor.use_count()) //this scorer exist
            {
              vps->scorers.push_back(scor);
            }
            else
            {
              scor = scorerFactory.createScorer(info.GetValue(), volume.GetUnplacedVolume()->Capacity() );
              m_globelScorers[info.GetValue()]=scor;
              vps->scorers.push_back(scor);
              std::cout << "vol name " << volume.GetName() <<" capacity "<<  volume.GetUnplacedVolume()->Capacity()  << std::endl;

            }
            std::cout << "vol name " << volume.GetName() <<" type "<< info.GetType() << " value " << info.GetValue() << std::endl;
          }
          else if(info.GetType() == "MirrorPhysics")
          {
            std::cout << "vol name " << volume.GetName() <<" type "<< info.GetType() << " value " << info.GetValue() << std::endl;
            std::cout << "info is " << info.GetValue() << std::endl;

            vps->mirrorPhysics = std::make_shared<MirrorPhyiscs>(std::stod(info.GetValue()), 1e-5);
            std::cout << "added mirror physics " << std::endl;
          }
        }
      }
    }

    // 3. setup physics model, if it is not yet set
    const vgdml::Material& mat = mat_iter->second;
    auto matphys = getMaterialPhysics(mat.name);

    if(m_logVolID2Mateiral.find(volID)==m_logVolID2Mateiral.end())
    {
      m_logVolID2Mateiral.insert( std::make_pair<int, std::string>(volID , std::string(mat.attributes.find("atomValue")->second )) );
    }

    if(matphys) //m_logVolID2physcorer not exist
    {
      vps->physics=matphys;
      std::cout << "Set model " << mat.name
                << " for volume " << volume.GetName() << std::endl;
    }
    else
    {
      std::cout << "Creating model " << mat.name << ", "
                << mat.attributes.find("atomValue")->second << volume.GetName() << std::endl;
      std::shared_ptr<MaterialPhysics> model = std::make_shared<MaterialPhysics>();
      m_globelPhysics.insert( std::make_pair<std::string, std::shared_ptr<MaterialPhysics>>(std::string(mat.name) , std::move(model) ) );

      auto theNewPhysics = getMaterialPhysics(mat.name);
      double bias (1.);
      auto itbias = mat.attributes.find("D");

      if(itbias!=mat.attributes.end())
      {
        bias = std::stod(itbias->second);
      }
      const std::string &cfg = mat.attributes.find("atomValue")->second;
      theNewPhysics->addComposition(cfg, bias);
      vps->physics=theNewPhysics;
    }

    vps->sortScorers();

    std::cout << "volume name " << volume.GetName() << " (id = " << volume.id() << "): material name " << mat.name << std::endl;
    if (mat.attributes.size()) std::cout << "  attributes:\n";
    for (const auto &attv : mat.attributes)
      std::cout << "    " << attv.first << ": " << attv.second << std::endl;
    if (mat.fractions.size()) std::cout << "  fractions:\n";
    for (const auto &attv : mat.fractions)
      std::cout << "    " << attv.first << ": " << attv.second << std::endl;
    if (mat.components.size()) std::cout << "  components:\n";
    for (const auto &attv : mat.components)
      std::cout << "    " << attv.first << ": " << attv.second << std::endl;
  }
}

void Prompt::VolumePhysicsScorer::sortScorers()
{
  entry_scorers.clear();
  propagate_scorers.clear();
  exit_scorers.clear();
  surface_scorers.clear();
  absorb_scorers.clear();


  for(auto &v : scorers)
  {
    auto type = v->getType();
    if(type==Scorer::ENTRY)
    {
      entry_scorers.push_back(v);
      std::cout << "Added ENTRY type scorer: " << v->getName() << std::endl;
    }
    else if(type==Scorer::PROPAGATE)
    {
      propagate_scorers.push_back(v);
      std::cout << "Added PROPAGATE type scorer: " << v->getName() << std::endl;
    }
    else if(type==Scorer::EXIT)
    {
      exit_scorers.push_back(v);
      std::cout << "Added EXIT type scorer: " << v->getName() << std::endl;
    }
    else if(type==Scorer::SURFACE)
    {
      surface_scorers.push_back(v);
      std::cout << "Added SURFACE type scorer: " << v->getName() << std::endl;
    }
    else if(type==Scorer::ABSORB)
    {
      absorb_scorers.push_back(v);
      std::cout << "Added ABSORB type scorer: " << v->getName() << std::endl;
    }
    else
      PROMPT_THROW2(BadInput, "unknown scorer type " << type);
  }
}
