#include "Data.hh"


void* Data_new()
{
  return static_cast<void*>(new Data<double>());
}
void Data_delete(void* d)
{
  delete static_cast<Data<double> *>(d);
}

size_t Data_getDim(void* d)
{
  return static_cast<Data<double> *>(d)->dim.size();
}

long long unsigned* Data_getShape(void* d)
{
  return static_cast<Data<double> *>(d)->dim.data();
}

double* Data_getVec(void *d)
{
  return static_cast<Data<double> *>(d)->vec.data();
}

void Data_sanityCheck(void* d)
{
  static_cast<Data<double> *>(d)->sanityCheck();
}
