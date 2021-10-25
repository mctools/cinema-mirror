#ifndef HistCtypes_hh
#define HistCtypes_hh

#ifdef __cplusplus
extern "C" {
#endif
  //HistBase
  void HistBase_save(void* obj, char *fn);
  void HistBase_scale(void* obj, double factor);
  const double* HistBase_getRaw(void* obj);
  const double* HistBase_getHit(void* obj);
  unsigned HistBase_getNBin(void* obj);


  //Hist1D
  void* Hist1D_new(double xmin, double xmax, unsigned nxbins, bool log);
  void Hist1D_delete(void* obj);
  void Hist1D_fill(void* obj, unsigned n, double* xval);
  void Hist1D_fillWeighted(void* obj,unsigned n, double* xval, double* weight);

  //Hist2D
  void* Hist2D_new(double xmin, double xmax, unsigned nxbins,
               double ymin, double ymax, unsigned nybins);
  void Hist2D_delete(void* obj);
  void Hist2D_fill(void* obj, unsigned n, double* xval, double* yval);
  void Hist2D_fillWeighted(void* obj,unsigned n, double* xval, double* yval, double* weight);
  unsigned Hist2D_getNBinX(void* obj);
  unsigned Hist2D_getNBinY(void* obj);

  #ifdef __cplusplus
  }
  #endif


#endif
