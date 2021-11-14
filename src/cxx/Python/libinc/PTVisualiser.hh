#ifndef Prompt_Visualiser_hh
#define Prompt_Visualiser_hh

#include "PromptCore.hh"

#ifdef __cplusplus
extern "C" {
#endif

size_t prompt_pVolSize();
void prompt_meshInfo(size_t pvolID, unsigned nSegments, size_t &npoints, size_t &nPlolygen);
void prompt_getMesh(size_t pvolID, unsigned nSegments, double *points, size_t *NumPolygonPoints);
void prompt_printMesh();



#ifdef __cplusplus
}
#endif
#endif
