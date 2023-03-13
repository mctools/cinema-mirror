#ifndef Prompt_VecGeom_hh
#define Prompt_VecGeom_hh

#ifdef __cplusplus
extern "C" {
#endif

void pt_setWorld(void* logicalWorld);

// Box
void* pt_Box_new(double hx, double hy, double hz);
void pt_Box_delete(void* obj);

void *pt_Tessellated_new(double x1, double x2, double y1, double y2, double z);


// Volume 
void* pt_Volume_new(const char* name, void *unplacedVolume);
void pt_Volume_delete(void* obj);
void pt_Volume_placeChild(void* obj, const char* name, void *Volume, void *transformation, int group);
unsigned pt_Volume_id(void* obj);
// unsigned pt_Volume_copyid(void *obj):

#ifdef __cplusplus
}
#endif

#endif
