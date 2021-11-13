#include "PTVisualiser.hh"
#include <VecGeom/base/Config.h>
#include <VecGeom/management/GeoManager.h>
#include <VecGeom/gdml/Middleware.h>
#include <VecGeom/navigation/BVHNavigator.h>
#include <VecGeom/navigation/NewSimpleNavigator.h>
#include <VecGeom/gdml/Frontend.h>

void prompt_placedVolume(double *points, unsigned *numPointInFace, unsigned *faces)
{
  auto &geoManager = vecgeom::GeoManager::Instance();
  size_t pvolsize =  geoManager.GetPlacedVolumesCount();
  for (size_t i=0; i<pvolsize; i++)
  {
    // const vgdml::VPlacedVolume
    auto *vol = geoManager.Convert(i);
    std::cout << vol->GetName() << std::endl;

    // Utils3D::USolidMesh
    auto *mesh = vol->CreateMesh3D(10);
    // std::vector<double> points,
    // std::vector<unsigned> numPointInFace,
    // std::vector<unsigned> faces
    for(const auto &v: mesh->GetPolygons())
    {
      std::cout << v << std::endl;
      *(++numPointInFace) = v.fN;
      for (size_t i = 0; i < v.fN; ++i)
      {
        *(faces++)=i;
        *(points++)=v.GetVertex(i)[0];
        *(points++)=v.GetVertex(i)[1];
        *(points++)=v.GetVertex(i)[2];
        std::cout <<  v.GetVertex(i) << "\n";
      }
    }
  }
}
