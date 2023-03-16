#include "PTVecGeom.hh"
#include "PromptCore.hh"

#include "VecGeom/management/GeoManager.h"
#include "VecGeom/volumes/UnplacedBox.h"
#include "VecGeom/volumes/UnplacedVolume.h"
#include "VecGeom/volumes/SpecializedTessellated.h"
#include "VecGeom/volumes/UnplacedTessellated.h"
#include "VecGeom/volumes/MultiUnion.h"

#include "VecGeom/navigation/BVHNavigator.h"
#include "VecGeom/navigation/NewSimpleNavigator.h"
#include "VecGeom/navigation/SimpleABBoxNavigator.h"
#include "VecGeom/navigation/SimpleABBoxLevelLocator.h"
#include "VecGeom/navigation/BVHLevelLocator.h"

namespace vg = VECGEOM_NAMESPACE;


void pt_setWorld(void* logicalWorld)
{
    vg::GeoManager::Instance().SetWorld(static_cast<vg::LogicalVolume *>(logicalWorld)->Place());
    vg::GeoManager::Instance().CloseGeometry();
      //accelaration
    vecgeom::BVHManager::Init();
    for (auto &lvol : vecgeom::GeoManager::Instance().GetLogicalVolumesMap()) {
        auto ndaughters = lvol.second->GetDaughtersp()->size();

        if (ndaughters <= 2)
        lvol.second->SetNavigator(vecgeom::NewSimpleNavigator<>::Instance());
        else
        lvol.second->SetNavigator(vecgeom::BVHNavigator<>::Instance());

        if (lvol.second->ContainsAssembly())
        lvol.second->SetLevelLocator(vecgeom::SimpleAssemblyAwareABBoxLevelLocator::GetInstance());
        else
        lvol.second->SetLevelLocator(vecgeom::BVHLevelLocator::GetInstance());
    }
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


void *pt_Tessellated_new(size_t faceVecSize, size_t* faces, float *point)
{
  // Create a tessellated solid from Trd parameters
//   vg::SimpleTessellated *stsl  = new vg::SimpleTessellated("pt_Tessellated_new");
//   vg::UnplacedTessellated *tsl = (vg::UnplacedTessellated *)stsl->GetUnplacedVolume();

    vg::UnplacedTessellated *tsl = new vg::UnplacedTessellated ();

    for(size_t i=0;i<faceVecSize;i++)
    {
        auto nVert = (*(faces++));
        if(nVert==3)
        {
            tsl->AddTriangularFacet(*reinterpret_cast<const vg::Vector3D<float>*>((point+ (*(faces++))*3)),
            *reinterpret_cast<const vg::Vector3D<float>*>(point+ (*(faces++))*3),
            *reinterpret_cast<const vg::Vector3D<float>*>(point+ (*(faces++))*3));
            i += 3;
        }
        else if(nVert==4)
        {
            tsl->AddQuadrilateralFacet(*reinterpret_cast<const vg::Vector3D<float>*>(point+(*(faces++))*3),
            *reinterpret_cast<const vg::Vector3D<float>*>(point+(*(faces++))*3),
            *reinterpret_cast<const vg::Vector3D<float>*>(point+(*(faces++))*3),
            *reinterpret_cast<const vg::Vector3D<float>*>(point+(*(faces++))*3));
            i += 4;
        }
        else
        {
            std::cout << *(faces++) << " "
            << *(faces++) << " "
            << *(faces++) << " "
            << *(faces++) << " "
            << *(faces++) << "\n";
            PROMPT_THROW2(CalcError, "the face should constain either 3 or 4 vert " 
                << faceVecSize << " " << i << " " << nVert);
        }
    }
    tsl->Close();

    return static_cast<void *>(tsl);
}


// Volume 
void* pt_Volume_new(const char* name, void *unplacedVolume)
{
    auto p = static_cast<void *>(new vg::LogicalVolume(name, static_cast<vg::VUnplacedVolume *>(unplacedVolume)));
    const std::map<unsigned int, vg::LogicalVolume *> & vmap  = vg::GeoManager::Instance().GetLogicalVolumesMap();
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

