#include "PTVisualiser.hh"
#include <VecGeom/base/Config.h>
#include <VecGeom/management/GeoManager.h>
#include <VecGeom/gdml/Middleware.h>
#include <VecGeom/navigation/BVHNavigator.h>
#include <VecGeom/navigation/NewSimpleNavigator.h>
#include <VecGeom/gdml/Frontend.h>

void prompt_placedVolume()
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
    // // mesh->GetVertices() returns Utils3D::vector_t<Utils3D::Vec_t>
    // for(const auto &v: mesh->GetVertices())
    // {
    //   std::cout << v << std::endl;
    // }
    // mesh->GetPolygons() returns Utils3D::vector_t<Utils3D::Polygon>
    std::vector<double> points;
    std::vector<unsigned> faces;
    for(const auto &v: mesh->GetPolygons())
    {
      std::cout << v << std::endl;
      for (size_t i = 0; i < v.fN; ++i)
      {
        faces.push_back(i);
        points.push_back(v.GetVertex(i)[0]);
        points.push_back(v.GetVertex(i)[1]);
        points.push_back(v.GetVertex(i)[2]);
        std::cout <<  v.GetVertex(i) << "\n";
      }
    }
  }
}
