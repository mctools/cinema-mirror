#include <string>

#include <VecGeom/base/Config.h>
#include <VecGeom/management/GeoManager.h>
#include <VecGeom/gdml/Frontend.h>


int main(int argc, char** argv)
{
  std::string gdml_name=(argv[1]);
  bool load = vgdml::Frontend::Load(gdml_name.c_str(), false, 1);
   if (!load) return 3;


  const vecgeom::VPlacedVolume *world = vecgeom::GeoManager::Instance().GetWorld();
   if (!world) return 4;
}
