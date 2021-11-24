#include "PTMeshHelper.hh"
#include <VecGeom/base/Config.h>
#include <VecGeom/volumes/SolidMesh.h>
#include <VecGeom/management/GeoManager.h>
#include <VecGeom/gdml/Middleware.h>
#include <VecGeom/navigation/BVHNavigator.h>
#include <VecGeom/navigation/NewSimpleNavigator.h>
#include <VecGeom/gdml/Frontend.h>

size_t pt_placedVolNum()
{
  auto &geoManager = vecgeom::GeoManager::Instance();
  return geoManager.GetPlacedVolumesCount();
}

void pt_meshInfo(size_t pvolID, size_t nSegments, size_t &npoints, size_t &nPlolygen, size_t &faceSize)
{
  auto &geoManager = vecgeom::GeoManager::Instance();
  size_t pvolsize =  geoManager.GetPlacedVolumesCount();

  // const vgdml::VPlacedVolume
  auto *vol = geoManager.Convert(pvolID);

  // Utils3D::USolidMesh
  auto *mesh = vol->CreateMesh3D(nSegments);
  if(!mesh)
  {
    npoints=0;
    nPlolygen=0;
    faceSize=0;
    return;
  }

  nPlolygen = mesh->GetPolygons().size();
  npoints = 0;
  for(const auto &apolygon: mesh->GetPolygons())
  {
    if(!npoints)
      npoints = apolygon.fVert.size();
    faceSize += apolygon.fInd.size();
  }
}

const char* pt_getMeshName(size_t pvolID)
{
  auto &geoManager = vecgeom::GeoManager::Instance();
  return geoManager.Convert(pvolID)->GetName();
}

//size of points: 3*n
//size of faces: n
//size of NumPolygonPoints: m
void pt_getMesh(size_t pvolID, size_t nSegments, double *points, size_t *NumPolygonPoints, size_t *faces)
{
  auto &geoManager = vecgeom::GeoManager::Instance();

  // const vgdml::VPlacedVolume
  auto *vol = geoManager.Convert(pvolID);
  // std::cout << vol->GetName() << std::endl;
  // Utils3D::USolidMesh
  auto *mesh = vol->CreateMesh3D(nSegments);

   if(mesh->GetPolygons().empty())
      PROMPT_THROW(BadInput, "empty mesh");


   auto &polygens = mesh->GetPolygons();
   for(const auto &poly: polygens)
   {
     for(const auto &vert: poly.fVert)
     {
       *(points++)=vert[0];
       *(points++)=vert[1];
       *(points++)=vert[2];
     }
     break;
   }

  for(const auto &apolygon: polygens)
  {
    *(NumPolygonPoints++) = apolygon.fN;
    *(faces++)  = apolygon.fN;

    for(auto vt : apolygon.fInd )
    {
      *(faces++) = vt;
      // std::cout << vt << std::endl;
    }
  }
}

void pt_printMesh()
{
  std::vector<vecgeom::VPlacedVolume *> v1;
  vecgeom::GeoManager::Instance().getAllPlacedVolumes(v1);
  for (auto &plvol : v1)
  {
    std::cerr << "placedVol=" << plvol << ", name=" << plvol->GetName()
              << ". type name " << typeid(plvol).name()
              << ". Number of solids " << plvol->GetDaughters().size()
              << ", " << vecgeom::GeoManager::Instance().GetWorld() << ">\n";

    // Utils3D::USolidMesh
    auto *mesh = plvol->CreateMesh3D(10);
    if(mesh)
    {
      for(const auto &v: mesh->GetPolygons())
      {
        std::cout << v << std::endl;
      }
    }
  }

  // auto &geoManager = vecgeom::GeoManager::Instance();
  // size_t pvolsize =  geoManager.GetPlacedVolumesCount();
  // for (size_t i=0; i<pvolsize; i++)
  // {
  //   // const vgdml::VPlacedVolume
  //   auto *vol = geoManager.Convert(i);
  //   std::cout << vol->GetName() << std::endl;
  //   // Utils3D::USolidMesh
  //   auto *mesh = vol->CreateMesh3D(10);
  //
  //   for(const auto &v: mesh->GetPolygons())
  //   {
  //     std::cout << v << std::endl;
  //   }
  // }
}
