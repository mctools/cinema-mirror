#ifndef Prompt_VecGeom_hh
#define Prompt_VecGeom_hh

#ifdef __cplusplus
extern "C" {
#endif


void pt_setWorld(void* logicalWorld);

// UnplacedBox
void* pt_UnplacedBox_new(double hx, double hy, double hz);
void pt_UnplacedBox_delete(void* obj);


// LogicalVolume 
void* pt_LogicalVolume_new(const char* name, void *unplacedVolume);
void pt_LogicalVolume_delete(void* obj);
void pt_LogicalVolume_placeDaughter(void* obj, const char* name, void *logicalVolume, void *transformation);
unsigned pt_LogicalVolume_id(void* obj);

#ifdef __cplusplus
}
#endif

#endif
