#ifndef Hdf5Trj_hh
#define Hdf5Trj_hh

#include "Trajectory.hh"
#include "hdf5.h"


class Hdf5Trj : public Trajectory {
public:
  Hdf5Trj(std::string fileName);
  virtual ~Hdf5Trj();

  virtual void readAtomTrj(unsigned atomOffset, std::vector<double> &trj) const override;
  virtual void readFrame(unsigned frameID, std::vector<double> &frame) const override;
  virtual const double* const readAtomTrj(unsigned atomOffset) const override;

  void cacheAtomTrj();
  void clearCache();

private:
  hid_t m_file_id;
  Data<double> m_fullTrj;
  template <class T>
  void readDataset(const std::string &name, Data<T> &vec, bool loadData=true);

};

#ifdef __cplusplus
extern "C" {
#endif

  void* Hdf5Trj_new(const char* fileName);
  void Hdf5Trj_delete(void* self);
  double Hdf5Trj_getDeltaT(void* self);
  double Hdf5Trj_getQResolution(void* self);
  void Hdf5Trj_frequency(void* self, void *fre );
  void Hdf5Trj_vDosSqw(void* self, unsigned firstAtomPos, void *vdos, void *omega,
                 void*Q , void *sqw);
  void Hdf5Trj_structFactSqw(void* self, unsigned qSize,
                  void *structFact, void *sqw);
#ifdef __cplusplus
}
#endif

#endif
