#ifndef Prompt_PythonResource_hh
#define Prompt_PythonResource_hh

#include "PTScorer.hh"

#ifdef __cplusplus
extern "C" {
#endif
 
void pt_ResourceManager_addNewVolume(unsigned volID);
void pt_ResourceManager_addScorer(unsigned volID, const char* cfg, void* scorer);
void pt_ResourceManager_addSurface(unsigned volID, const char* cfg);
void pt_ResourceManager_cfgVolPhysics(unsigned volID, const char* cfg);
void* pt_ResourceManager_getHist(const char* cfg);
void pt_ResourceManager_clear(); 

#ifdef __cplusplus
}
#endif

#endif
