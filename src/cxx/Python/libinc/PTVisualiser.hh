#ifndef Prompt_Visualiser_hh
#define Prompt_Visualiser_hh

#include "PromptCore.hh"

#ifdef __cplusplus
extern "C" {
#endif

size_t pt_placedVolNum();
const char* pt_getMeshName(size_t pvolID);
void pt_meshInfo(size_t pvolID, size_t nSegments, size_t &npoints, size_t &nPlolygen, size_t &faceSize);
void pt_getMesh(size_t pvolID, size_t nSegments, double *points, size_t *NumPolygonPoints, size_t *faces);
void pt_printMesh();

#ifdef __cplusplus
}
#endif
#endif
