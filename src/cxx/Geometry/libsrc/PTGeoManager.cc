#include "PTGeoManager.hh"

#include <VecGeom/base/Config.h>
#include <VecGeom/management/GeoManager.h>
#include <VecGeom/gdml/Middleware.h>
#include <VecGeom/gdml/Frontend.h>
#include <VecGeom/volumes/PlacedVolume.h>



#include "PTAnaManager.hh"

#include "PTUtils.hh"
#include "PTMaxwellianGun.hh"
#include "PTSimpleThermalGun.hh"
#include "PTNeutron.hh"

Prompt::GeoManager::GeoManager()
:m_gun(nullptr)
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
      else if(words[0]=="SimpleThermalGun")
      {
        double temp = std::stod(words[1]);
        m_gun = std::make_shared<SimpleThermalGun>(Neutron(), temp, string2vec(words[2]), string2vec(words[3]));
      }
      else
        PROMPT_THROW2(BadInput, "MaxwellianGun only for the moement");
    }
  }


  // Get the volume auxiliary info
  const std::map<int, std::vector<vgdml::Auxiliary>>& volAuxInfo = aMiddleware.GetVolumeAuxiliaryInfo();
  std::cout << "Geometry contains "
            << volAuxInfo.size() << " entries of volum auxiliary info\n";

  auto &anaManager = Singleton<AnaManager>::getInstance();
  auto &geoManager = vecgeom::GeoManager::Instance();

  //geoManager.GetLogicalVolumesMap() returens std::map<unsigned int, LogicalVolume *>
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


    // 1. filter out material-empty volume
    auto mat_iter = volumeMatMap.find(volID);
    if(mat_iter==volumeMatMap.end()) //union creates empty virtual volume
      PROMPT_THROW(CalcError, "empty volume ")
      // continue;

    // 2. setup scorors
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
          if (info.GetType() != "Sensitive")
            continue;

          std::shared_ptr<Prompt::Scoror> scor = getScoror(info.GetValue());

          if(scor.use_count()) //this scorer exist
          {
            vps->scorors.push_back(scor);
          }
          else
          {
            scor = anaManager.createScoror(info.GetValue());
            m_globelScorors[info.GetValue()]=scor;
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

    vps->sortScorors();

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
