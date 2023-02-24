#include "PTVecGeom.hh"


#include "VecGeom/management/GeoManager.h"
#include "VecGeom/volumes/Box.h"
#include "VecGeom/volumes/UnplacedVolume.h"

namespace vg = VECGEOM_NAMESPACE;


void pt_setWorld(void* logicalWorld)
{
    vg::GeoManager::Instance().SetWorld(static_cast<vg::LogicalVolume *>(logicalWorld)->Place());
    vg::GeoManager::Instance().CloseGeometry();
}

//   UnplacedBox *worldUnplaced      = new UnplacedBox(10, 10, 10)
void* pt_UnplacedBox_new(double hx, double hy, double hz)
{
    return static_cast<void *>(new vg::UnplacedBox(hx, hy, hz));
}

void pt_UnplacedBox_delete(void* obj)
{
    delete static_cast<vg::UnplacedBox *>(obj);
}


// LogicalVolume 
void* pt_LogicalVolume_new(const char* name, void *unplacedVolume)
{
    auto p = static_cast<void *>(new vg::LogicalVolume(name, static_cast<vg::VUnplacedVolume *>(unplacedVolume)));
    const std::map<unsigned int, vecgeom::LogicalVolume *> & vmap  = vecgeom::GeoManager::Instance().GetLogicalVolumesMap();
    for(auto it=vmap.begin(); it!=vmap.end(); ++it)
    {
        std::cout << "pt_LogicalVolume_new " << it->second->GetName() << ", vol id " << it->first << std::endl; 
    }
    std::cout << "\n";
    return p;
}

void pt_LogicalVolume_delete(void* obj)
{
    delete static_cast<vg::LogicalVolume *>(obj);
}

void pt_LogicalVolume_placeDaughter(void* obj, const char* name, void *logicalVolume, void *transformation)
{
    auto vol = static_cast<vg::LogicalVolume *>(logicalVolume);
    auto transf = static_cast<const vg::Transformation3D *>(transformation);
    static_cast<vg::LogicalVolume *>(obj)->PlaceDaughter(name, vol, transf);
}

unsigned pt_LogicalVolume_id(void* obj)
{
    return static_cast<vg::LogicalVolume *>(obj)->id();
}

