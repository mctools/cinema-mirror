#ifndef HistCtypes_hh
#define HistCtypes_hh

extern "C" {
  //NumpyHist2D
  void* NumpyHist2D_new(unsigned nxbins, double xmin, double xmax,
              unsigned nybins, double ymin, double ymax);
  void NumpyHist2D_delete(void* obj);
  void NumpyHist2D_fill(void* obj, unsigned n, double* xval, double* yval);
  void NumpyHist2D_fillWeighted(void* obj,unsigned n, double* xval, double* yval, double* weight);
  unsigned NumpyHist2D_getNBinX(void* obj);
  unsigned NumpyHist2D_getNBinY(void* obj);

  void NumpyHistBase_save(void* obj, char *fn);
  unsigned NumpyHistBase_getNBins(void* obj);
  const double* NumpyHistBase_getRaw(void* obj);
}

#endif
