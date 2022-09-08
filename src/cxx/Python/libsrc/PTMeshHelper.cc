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

#include "PTMeshHelper.hh"
#include <VecGeom/base/Config.h>
#include <VecGeom/volumes/SolidMesh.h>
#include <VecGeom/management/GeoManager.h>
#include <VecGeom/gdml/Middleware.h>
#include <VecGeom/navigation/BVHNavigator.h>
#include <VecGeom/navigation/NewSimpleNavigator.h>
#include <VecGeom/gdml/Frontend.h>

#include <VecGeom/base/Transformation3D.h>
#include "VecGeom/base/Vector3D.h"

class GeoTree
{
  struct Node {
    int physical, logical;
    vecgeom::Transformation3D *matrix;
    std::vector<std::shared_ptr<Node>> child;
  };

public:
  GeoTree();
  ~GeoTree();

  std::shared_ptr<Node> getRoot();
  std::shared_ptr<Node> findRootChild(int num);
  std::shared_ptr<Node> findChild(std::shared_ptr<Node> node, int num);

private:
  std::shared_ptr<Node> m_root;
};

GeoTree::GeoTree() {}

GeoTree::~GeoTree() {}

std::shared_ptr<GeoTree::Node> GeoTree::getRoot() { return m_root; }

std::shared_ptr<GeoTree::Node> GeoTree::findRootChild(int num)
{
  return findChild(m_root, num);
}

std::shared_ptr<GeoTree::Node> GeoTree::findChild(std::shared_ptr<Node> node, int num)
{
  if(!node)
    return nullptr;
  if (num == node->physical)
    return node;
  for (auto childptr : node->child)
	{
		return findChild(childptr, num);
	}
  	return nullptr;
}



void* pt_Transformation3D_new(void *consttrfm3Dobj)
{
  auto p = static_cast<const vecgeom::Transformation3D*> (consttrfm3Dobj) ;
  return static_cast<void *>(new vecgeom::Transformation3D(*p));
}

void* pt_Transformation3D_newfromID(int id)
{
  auto &geoManager = vecgeom::GeoManager::Instance();
  // const vgdml::VPlacedVolume
  auto *vol = geoManager.Convert(id);

  auto p = new vecgeom::Transformation3D(* const_cast<vecgeom::Transformation3D*>(vol->GetTransformation()));
  p->Inverse(*p); //NOTICE the inversion here, so the tranform function is from local to mater
  return static_cast<void *>(p);
}

void pt_Transformation3D_delete(void *trfm3Dobj)
{
  delete static_cast<vecgeom::Transformation3D *>(trfm3Dobj);
}

void pt_Transformation3D_multiple(void *trfm3Dobj1, void *trfm3Dobj2)
{
   auto obj1 = static_cast<vecgeom::Transformation3D *>(trfm3Dobj1);
   auto obj2 = static_cast<vecgeom::Transformation3D *>(trfm3Dobj2);
   obj1->MultiplyFromRight(*obj2);
}

void pt_Transformation3D_transform(void *obj, size_t numPt, double *in, double *out)
{
  auto mat = static_cast<vecgeom::Transformation3D *>(obj);
  for(size_t i=0;i<numPt;i++)
  {
    vecgeom::Vector3D<Precision> vert(in[i*3], in[i*3+1], in[i*3+2]);
    auto vertTransformed = mat->Transform(vert);
    // auto vertTransformed = mat->InverseTransform(vert);
    out[i*3] = vertTransformed[0];
    out[i*3+1] = vertTransformed[1];
    out[i*3+2] = vertTransformed[2];
  }
}

const char* pt_Transformation3D_print(void *trfm3Dobj)
{
  std::ostringstream a;
  static_cast<vecgeom::Transformation3D *>(trfm3Dobj)->Print(a);
  std::string str;
  a << str;
  return str.c_str();
}

size_t pt_placedVolNum()
{
  auto &geoManager = vecgeom::GeoManager::Instance();
  printf("total volume in the tree %ld\n", geoManager.GetTotalNodeCount());
  printf("total placed volume %ld\n", geoManager.GetPlacedVolumesCount());

  return geoManager.GetPlacedVolumesCount();
}

size_t pt_numDaughters(size_t pvolID)
{
  auto &geoManager = vecgeom::GeoManager::Instance();
  // const vgdml::VPlacedVolume
  auto *vol = geoManager.Convert(pvolID);
  // printf("\nfrom xx: vol name \"%s\", volID %zu, copy id %d, daughter num %zu\n", vol->GetName(), pvolID, vol->GetCopyNo(), vol->GetDaughters().size());
  printf("\nphysical volume info : %zu\n", pvolID);
  vol->Print();
  printf("\n");
  return vol->GetDaughters().size();
}

void pt_getDaughterID(size_t pvolID, size_t dsize, unsigned *physicalID, unsigned *logicalID)
{
  auto &geoManager = vecgeom::GeoManager::Instance();
  // const vgdml::VPlacedVolume
  auto *vol = geoManager.Convert(pvolID);
  auto &dau = vol->GetDaughters();
  if(dau.size() != dsize)
    PROMPT_THROW(BadInput, "Wrong daughter size");

  for(auto d: dau)
  {
    (*physicalID++)=d->id();
    (*logicalID++)=d->GetLogicalVolume()->id();
  }
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

   const vecgeom::Transformation3D *matrix = vol->GetTransformation();

   // auto = Utils3D::vector_t<Utils3D::Polygon> const
   auto &polygens = mesh->GetPolygons();
   for(const auto &poly: polygens)
   {
     for(const auto &vert: poly.fVert)
     {
       // auto vertTransformed = matrix->Transform(vert);
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
}
