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

size_t pt_Launcher_getTrajSize(void* obj);
void pt_Launcher_getTrajectory(void* obj, double *trj);
void pt_Launcher_go(void* obj, uint64_t numParticle, double printPrecent, bool recordTrj);

#ifdef __cplusplus
}
#endif

#endif
