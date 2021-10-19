#include "PTGeoManager.hh"

#include <VecGeom/base/Config.h>
#include <VecGeom/management/GeoManager.h>
#include <VecGeom/gdml/Middleware.h>
#include <VecGeom/navigation/BVHNavigator.h>
#include <VecGeom/navigation/NewSimpleNavigator.h>
#include <VecGeom/gdml/Frontend.h>

#include "PTNCrystal.hh"


Prompt::GeoManager::GeoManager(const std::string &gdml_file)
{
  // PTNCrystal();
  //access material
  vgdml::Parser p;
  double mm_unit = 0.1;
  const auto loadedMiddleware = p.Load(gdml_file.c_str(), false, mm_unit);

  if (!loadedMiddleware) PROMPT_THROW(DataLoadError, "failed to load the gdml file ");

  const auto &aMiddleware = *loadedMiddleware;
  auto volumeMatMap   = aMiddleware.GetVolumeMatMap();
  auto materialMap = aMiddleware.GetMaterialMap();

  for(const auto &item : materialMap)
  {
    // item->first name
    // item->second material

  }


  // //initialise material
  // using MaterialMap_t         = std::map<std::string, vgdml::Material>;
  // MaterialMap_t const &GetMaterialMap() const { return materialMap; }

  //initialise navigation
  auto &geoManager = vecgeom::GeoManager::Instance();
  auto navigator = vecgeom::BVHNavigator<>::Instance();

  for (const auto &item : geoManager.GetLogicalVolumesMap())
  {
    //fill volume into nativator
    auto &volume   = *item.second;
    auto nchildren = volume.GetDaughters().size();
    volume.SetNavigator(nchildren > 0 ? navigator : vecgeom::NewSimpleNavigator<>::Instance());
    auto mat_iter = volumeMatMap.find( volume.id());
    if(mat_iter==volumeMatMap.end())
      continue;
      // PROMPT_THROW2(DataLoadError, "the material of "<< volume.GetName() << " has invalid id " << volume.id());

    const vgdml::Material& mat = mat_iter->second;

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

Prompt::GeoManager::~GeoManager(){}
