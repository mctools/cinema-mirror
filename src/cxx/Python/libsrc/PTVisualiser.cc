#include "PTVisualiser.hh"
#include <VecGeom/base/Config.h>
#include <VecGeom/base/Utils3D.h>
#include <VecGeom/management/GeoManager.h>
#include <VecGeom/gdml/Middleware.h>
#include <VecGeom/navigation/BVHNavigator.h>
#include <VecGeom/navigation/NewSimpleNavigator.h>
#include <VecGeom/gdml/Frontend.h>

size_t prompt_pVolSize()
{
  auto &geoManager = vecgeom::GeoManager::Instance();
  return geoManager.GetPlacedVolumesCount();
}

void prompt_meshInfo(size_t pvolID, unsigned nSegments, size_t &npoints, size_t &nPlolygen)
{
  auto &geoManager = vecgeom::GeoManager::Instance();
  size_t pvolsize =  geoManager.GetPlacedVolumesCount();

  // const vgdml::VPlacedVolume
  auto *vol = geoManager.Convert(pvolID);
  std::cout << vol->GetName() << std::endl;

  // Utils3D::USolidMesh
  auto *mesh = vol->CreateMesh3D(nSegments);
  nPlolygen = mesh->GetPolygons().size();
  npoints = mesh->GetVertices().size();
}

//size of points: 3*n
//size of faces: n
//size of NumPolygonPoints: m
void prompt_getMesh(size_t pvolID, unsigned nSegments, double *points, size_t *NumPolygonPoints)
{
  auto &geoManager = vecgeom::GeoManager::Instance();
  size_t pvolsize =  geoManager.GetPlacedVolumesCount();

  // const vgdml::VPlacedVolume
  auto *vol = geoManager.Convert(pvolID);
  std::cout << vol->GetName() << std::endl;
  // Utils3D::USolidMesh
  auto *mesh = vol->CreateMesh3D(nSegments);
  for(const auto &v: mesh->GetPolygons())
  {
    std::cout << v << std::endl;
    *(++NumPolygonPoints) = v.fN;
    for (size_t i = 0; i < v.fN; ++i)
    {
      *(points++)=v.GetVertex(i)[0];
      *(points++)=v.GetVertex(i)[1];
      *(points++)=v.GetVertex(i)[2];
      std::cout <<  v.GetVertex(i) << "\n";
    }
  }
}

void prompt_printMesh()
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

    for(const auto &v: mesh->GetPolygons())
    {
      std::cout << v << std::endl;
    }
  }
}
