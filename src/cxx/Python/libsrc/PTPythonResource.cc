#include "PTPythonResource.hh"
#include "PTResourceManager.hh"

namespace pt = Prompt;

void pt_ResourceManager_addNewVolume(unsigned volID)
{
    pt::Singleton<pt::ResourceManager>::getInstance().addNewVolume(volID);
}

void pt_ResourceManager_addScorer(unsigned volID, const char* cfg)
{
    pt::Singleton<pt::ResourceManager>::getInstance().addScorer(volID, cfg);
}

void pt_ResourceManager_addSurface(unsigned volID, const char* cfg)
{
    pt::Singleton<pt::ResourceManager>::getInstance().addSurface(volID, cfg);
}

void pt_ResourceManager_addPhysics(unsigned volID, const char* cfg)
{
    pt::Singleton<pt::ResourceManager>::getInstance().addPhysics(volID, cfg);
}

void* pt_ResourceManager_getHist(const char* cfg)
{
    return static_cast<void*>(pt::Singleton<pt::ResourceManager>::getInstance().getHist(cfg));
}