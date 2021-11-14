#ifndef Prompt_Python_hh
#define Prompt_Python_hh

#include "PromptCore.hh"

#ifdef __cplusplus
extern "C" {
#endif

// Prompt::Launcher
void* pt_Launcher_getInstance();
void pt_Launcher_setSeed(void* obj, uint64_t seed);
void pt_Launcher_setGun(void* obj, void* objgun);
void pt_Launcher_loadGeometry(void* obj, const char* fileName);


#ifdef __cplusplus
}
#endif

#endif
