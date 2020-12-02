#include <string>

#include "HistCtypes.hh"
#include "NumpyHist2D.hh"
#include "NumpyHist1D.hh"
#include "NumpyHistBase.hh"


unsigned NumpyHist2D_getNBinX(void* obj)
{
  return static_cast<NumpyHist2D*>(obj)->getNBinX();
}

unsigned NumpyHist2D_getNBinY(void* obj)
{
  return static_cast<NumpyHist2D*>(obj)->getNBinY();
}

void* NumpyHist2D_new(unsigned nxbins, double xmin, double xmax,
            unsigned nybins, double ymin, double ymax)
{
  return static_cast<void*>(new NumpyHist2D(nxbins, xmin,  xmax,
               nybins,  ymin,  ymax));
}

void NumpyHist2D_delete(void* obj)
{
    delete static_cast<NumpyHist2D*>(obj);
}

void NumpyHist2D_fill(void* obj, unsigned n, double* xval, double* yval)
{
  auto hist=static_cast<NumpyHist2D*>(obj);
  for(unsigned i=0;i<n;i++)
    hist->fill(xval[i], yval[i]);
}

void NumpyHist2D_fillWeighted(void* obj, unsigned n, double* xval, double* yval, double* w)
{
  auto hist = static_cast<NumpyHist2D*>(obj);
  for(unsigned i=0;i<n;i++)
    hist->fill(xval[i], yval[i], w[i]);
}

void NumpyHistBase_save(void* obj, char *fn)
{
  static_cast<NumpyHistBase*>(obj)->save(std::string(fn));
}

const double* NumpyHistBase_getRaw(void* obj)
{
  return (static_cast<NumpyHistBase*>(obj)->getRaw()).data();
}

unsigned NumpyHistBase_getNBins(void* obj)
{
  return static_cast<NumpyHistBase*>(obj)->getNBin();
}
