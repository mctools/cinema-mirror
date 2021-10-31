#include "PTGeoManager.hh"

#include <VecGeom/base/Config.h>
#include <VecGeom/management/GeoManager.h>
#include <VecGeom/gdml/Middleware.h>
#include <VecGeom/navigation/BVHNavigator.h>
#include <VecGeom/navigation/NewSimpleNavigator.h>
#include <VecGeom/gdml/Frontend.h>

#include "PTNCrystal.hh"
#include "PTAnaManager.hh"

Prompt::GeoManager::GeoManager()
{
}

Prompt::GeoManager::~GeoManager()
{
  std::cout << "Simulation completed!\n";
  std::cout << "Simulation created " << numMaterialPhysics() << " material physics\n";
  std::cout << "There are " << numScoror() << " scorors in total\n";
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

std::shared_ptr<Prompt::Scoror> Prompt::GeoManager::getScoror(const std::string &name)
{
  auto it = m_globelScorors.find(name);
  if(it!= m_globelScorors.end())
  {
    return it->second;
  }
  else
    return nullptr;
}

void Prompt::GeoManager::loadFile(const std::string &gdml_file)
{
  vgdml::Parser p;
  const auto loadedMiddleware = p.Load(gdml_file.c_str(), false, 1.);

  if (!loadedMiddleware) PROMPT_THROW(DataLoadError, "failed to load the gdml file ");

  const auto &aMiddleware = *loadedMiddleware;
  auto volumeMatMap   = aMiddleware.GetVolumeMatMap();

  // Get the volume auxiliary info
  const std::map<int, std::vector<vgdml::Auxiliary>>& volAuxInfo = aMiddleware.GetVolumeAuxiliaryInfo();
  std::cout << "Geometry contains "
            << volAuxInfo.size() << " entries of volum auxiliary info\n";

  auto &anaManager = Singleton<AnaManager>::getInstance();

  //initialise navigator
  auto &geoManager = vecgeom::GeoManager::Instance();
  auto navigator = vecgeom::BVHNavigator<>::Instance();

  for (const auto &item : geoManager.GetLogicalVolumesMap())
  {
    auto &volume   = *item.second;
    const size_t volID = volume.id();
    std::shared_ptr<VolumePhysicsScoror> vps(nullptr);
    if(m_volphyscoror.find(volID)==m_volphyscoror.end())
    {
      m_volphyscoror.insert(std::make_pair(volID,  std::make_shared<VolumePhysicsScoror>()));
      vps = m_volphyscoror[volID];
    }
    else
    {
      PROMPT_THROW2(CalcError, "volume ID " << volID << " appear more than once")
    }


    // 1. fill volume into nativator
    auto nchildren = volume.GetDaughters().size();
    volume.SetNavigator(nchildren > 0 ? navigator : vecgeom::NewSimpleNavigator<>::Instance());
    auto mat_iter = volumeMatMap.find(volID);
    if(mat_iter==volumeMatMap.end()) //union creates empty virtual volume
      continue;

    // 2. setup scorors
    if(volAuxInfo.size())
    {
      auto volAuxInfo_iter = volAuxInfo.find(volID);
      if(volAuxInfo_iter != volAuxInfo.end()) //it volume contains a scorors
      {
        std::cout << volume.GetName()<< ", volID " << volID << " contains volAuxInfo\n";
        const std::vector<vgdml::Auxiliary> &volAuxInfoVec = (*volAuxInfo_iter).second;
        auto volAuxInfoSize = volAuxInfoVec.size();
        std::cout << "volAuxInfoSize " << volume.GetName()  << " " << volAuxInfoSize << std::endl;

        for(const auto& info : volAuxInfoVec)
        {
          auto scor = getScoror(info.GetType());

          if(scor) //this scorer exist
          {
            vps->scorors.push_back( scor);
          }
          else
          {
            scor = anaManager.createScoror(info.GetValue());
            m_globelScorors[info.GetType()]=scor;
            vps->scorors.push_back(scor);
          }
          std::cout << "vol name " << volume.GetName() <<" type "<< info.GetType() << " value " << info.GetValue() << std::endl;
        }
      }
    }

    // 3. setup physics model, if it is not yet set
    const vgdml::Material& mat = mat_iter->second;
    auto matphys = getMaterialPhysics(mat.name);
    if(matphys) //m_volphyscoror not exist
    {
      vps->physics=matphys;
      std::cout << "Set model " << mat.name
                << " for volume " << volume.GetName() << std::endl;
    }
    else
    {
      std::cout << "Creating model " << mat.name << ", "
                << mat.attributes.find("atomValue")->second << " for volume " << volume.GetName() << std::endl;
      std::shared_ptr<MaterialPhysics> model = std::make_shared<MaterialPhysics>();
      m_globelPhysics.insert( std::make_pair<std::string, std::shared_ptr<MaterialPhysics>>
                (std::string(mat.name) , std::move(model) ) );
      auto theNewPhysics = getMaterialPhysics(mat.name);
      const std::string &cfg = mat.attributes.find("atomValue")->second;
      theNewPhysics->addComposition(cfg);
      vps->physics=theNewPhysics;
    }

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
  vecgeom::BVHManager::Init();
}
