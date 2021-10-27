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

Prompt::GeoManager::~GeoManager(){}


void setLogicalVolumePhysicsScoror(vecgeom::LogicalVolume &lv, std::unique_ptr<Prompt::VolumePhysicsScoror> &vps)
{
  lv.SetUserExtensionPtr((void *)(vps.get()));
}


void Prompt::GeoManager::loadFile(const std::string &gdml_file)
{
  vgdml::Parser p;
  const auto loadedMiddleware = p.Load(gdml_file.c_str(), false, 1.);

  if (!loadedMiddleware) PROMPT_THROW(DataLoadError, "failed to load the gdml file ");

  const auto &aMiddleware = *loadedMiddleware;
  auto volumeMatMap   = aMiddleware.GetVolumeMatMap();

  // Get the volume auxiliary info
  const auto& volAuxInfo = aMiddleware.GetVolumeAuxiliaryInfo();
  std::cout << "Geometry contains "
            << volAuxInfo.size() << " entries of volum auxiliary info\n";

  auto &anaManager = Singleton<AnaManager>::getInstance();

  //initialise navigator
  auto &geoManager = vecgeom::GeoManager::Instance();
  auto navigator = vecgeom::BVHNavigator<>::Instance();

  for (const auto &item : geoManager.GetLogicalVolumesMap())
  {
    std::unique_ptr<VolumePhysicsScoror> vps = std::make_unique<VolumePhysicsScoror>();

    // 1. fill volume into nativator
    auto &volume   = *item.second;
    const size_t volID = volume.id();
    auto nchildren = volume.GetDaughters().size();
    volume.SetNavigator(nchildren > 0 ? navigator : vecgeom::NewSimpleNavigator<>::Instance());
    auto mat_iter = volumeMatMap.find(volID);
    if(mat_iter==volumeMatMap.end()) //union creates empty virtual volume
      continue;

    // 2. setup scorors
    if(volAuxInfo.size())
    {
      auto iter = volAuxInfo.find(volID);
      if(iter != volAuxInfo.end())
      {
        const auto &volAuxInfoVec = (*iter).second;
        auto volAuxInfoSize = volAuxInfoVec.size();
        std::cout << "volAuxInfoSize " << volume.GetName()  << " " << volAuxInfoSize << std::endl;

        for(const auto& info : volAuxInfoVec)
        {
          vps->scorors.emplace_back(std::move(anaManager.createScoror(info.GetValue())));
          std::cout <<"type "<< info.GetType() << " value " << info.GetValue() << std::endl;
        }
      }
    }

    // 3. setup physics model, if it is not yet set
    const vgdml::Material& mat = mat_iter->second;
    auto volmat_iter = m_volphyscoror.find(mat.name);
    if(volmat_iter==m_volphyscoror.end())
    {
      std::unique_ptr<MaterialPhysics> model = std::make_unique<MaterialPhysics>();
      const std::string &cfg = mat.attributes.find("atomValue")->second;
      model->addComposition(cfg);
      vps->physics=std::move(model);
      setLogicalVolumePhysicsScoror(volume, vps);
    }
    else
      // 3.1 link volum with physics using the void pointer in the volume
      setLogicalVolumePhysicsScoror(volume, m_volphyscoror[mat.name]);

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

    m_volphyscoror.insert( std::pair<std::string, std::unique_ptr<VolumePhysicsScoror> > (mat.name, std::move(vps)) );

  }
  vecgeom::BVHManager::Init();
}
