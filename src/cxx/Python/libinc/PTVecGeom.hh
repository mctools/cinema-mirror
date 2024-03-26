#ifndef Prompt_VecGeom_hh
#define Prompt_VecGeom_hh

#ifdef __cplusplus
#include <cstddef>
extern "C" {
#endif

void pt_initNavigators(bool use_bvh_navigator);
void pt_setWorld(void* logicalWorld);

// Box
void* pt_Box_new(double hx, double hy, double hz);
void pt_Box_delete(void* obj);


// Tube
void* pt_Tube_new(double rmin, double rmax, double z, double deltaphi, double startphi);

// Sphere
void* pt_Sphere_new(double rmin, double rmax, double startphi, double deltaphi , double starttheta, double deltatheta);

//Trapezoid
void* pt_Trapezoid_new(double x1, double x2, double y1, double y2, double z);


// Tessellated
void *pt_Tessellated_new(size_t faceVecSize, size_t* faces, float *point);

// Polyhedron
void *pt_Polyhedron_new(double phiStart, double phiDelta, const int sideCount, const int zPlaneCount,
                     double *zPlanes, double *rMin, double *rMax);

// Arbitrary_trapezoid
void *pt_ArbTrapezoid_new(double (*v11), double (*v12), double (*v13), double (*v14), 
                        double (*v21), double (*v22), double (*v23), double (*v24),
                        double halfHeight);


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
