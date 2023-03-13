#include "PTVecGeom.hh"


#include "VecGeom/management/GeoManager.h"
#include "VecGeom/volumes/UnplacedBox.h"
#include "VecGeom/volumes/UnplacedVolume.h"
#include "VecGeom/volumes/SpecializedTessellated.h"
#include "VecGeom/volumes/UnplacedTessellated.h"

namespace vg = VECGEOM_NAMESPACE;


void pt_setWorld(void* logicalWorld)
{
    vg::GeoManager::Instance().SetWorld(static_cast<vg::LogicalVolume *>(logicalWorld)->Place());
    vg::GeoManager::Instance().CloseGeometry();
}

//   Box *worldUnplaced      = new UnplacedBox(10, 10, 10)
void* pt_Box_new(double hx, double hy, double hz)
{
    return static_cast<void *>(new vg::UnplacedBox(hx, hy, hz));
}

void pt_Box_delete(void* obj)
{
    delete static_cast<vg::UnplacedBox *>(obj);
}


void *pt_Tessellated_new(double x1, double x2, double y1, double y2,
                                                     double z)
{
  // Create a tessellated solid from Trd parameters
  vg::SimpleTessellated *stsl  = new vg::SimpleTessellated("pt_Tessellated_new");
  vg::UnplacedTessellated *tsl = (vg::UnplacedTessellated *)stsl->GetUnplacedVolume();
  // Top facet
  tsl->AddQuadrilateralFacet(vg::Vector3D(-x2, y2, z), vg::Vector3D(-x2, -y2, z), vg::Vector3D(x2, -y2, z), vg::Vector3D(x2, y2, z));
  // Bottom facet
  tsl->AddQuadrilateralFacet(vg::Vector3D(-x1, y1, -z), vg::Vector3D(x1, y1, -z), vg::Vector3D(x1, -y1, -z), vg::Vector3D(-x1, -y1, -z));
  // Front facet
  tsl->AddQuadrilateralFacet(vg::Vector3D(-x2, -y2, z), vg::Vector3D(-x1, -y1, -z), vg::Vector3D(x1, -y1, -z), vg::Vector3D(x2, -y2, z));
  // Right facet
  tsl->AddQuadrilateralFacet(vg::Vector3D(x2, -y2, z), vg::Vector3D(x1, -y1, -z), vg::Vector3D(x1, y1, -z), vg::Vector3D(x2, y2, z));
  // Behind facet
  tsl->AddQuadrilateralFacet(vg::Vector3D(x2, y2, z), vg::Vector3D(x1, y1, -z), vg::Vector3D(-x1, y1, -z), vg::Vector3D(-x2, y2, z));
  // Left facet
  tsl->AddQuadrilateralFacet(vg::Vector3D(-x2, y2, z), vg::Vector3D(-x1, y1, -z), vg::Vector3D(-x1, -y1, -z), vg::Vector3D(-x2, -y2, z));
  tsl->Close();
  return static_cast<void *>(stsl);
}


// Volume 
void* pt_Volume_new(const char* name, void *unplacedVolume)
{
    auto p = static_cast<void *>(new vg::LogicalVolume(name, static_cast<vg::VUnplacedVolume *>(unplacedVolume)));
    const std::map<unsigned int, vecgeom::LogicalVolume *> & vmap  = vecgeom::GeoManager::Instance().GetLogicalVolumesMap();
    for(auto it=vmap.begin(); it!=vmap.end(); ++it)
    {
        std::cout << "pt_Volume_new " << it->second->GetName() << ", vol id " << it->first << std::endl; 
    }
    std::cout << "\n";
    return p;
}

void pt_Volume_delete(void* obj)
{
    delete static_cast<vg::LogicalVolume *>(obj);
}

void pt_Volume_placeChild(void* obj, const char* name, void *volume,
                                    void *transformation, int group)
{
    auto vol = static_cast<vg::LogicalVolume *>(volume);
    auto transf = static_cast<const vg::Transformation3D *>(transformation);
    auto constplaced = static_cast<vg::LogicalVolume *>(obj)->PlaceDaughter(name, vol, transf);
    vg::VPlacedVolume* placed = const_cast<vg::VPlacedVolume *>(constplaced);;

    if(group)
    {
        placed->SetCopyNo(group);
        std::cout << name << " Group ID " << placed->GetCopyNo() << std::endl;
    }
}

unsigned pt_Volume_id(void* obj)
{
    return static_cast<vg::LogicalVolume *>(obj)->id();
}



// unsigned pt_LogicalVolume_copyid(void* obj)
// {
//     return static_cast<vg::LogicalVolume *>(obj)->GetCopyNo()();
// }

