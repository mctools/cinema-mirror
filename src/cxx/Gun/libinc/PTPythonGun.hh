#ifndef Prompt_PythonGun_hh
#define Prompt_PythonGun_hh

#include "PTPrimaryGun.hh"
#include <Python.h>

namespace Prompt {
  class PythonGun : public PrimaryGun {
  public:
    PythonGun(PyObject *obj);
    virtual ~PythonGun(); 
    virtual std::unique_ptr<Particle> generate() override;
    virtual void sampleEnergy(double &ekin) override 
    { PROMPT_THROW(CalcError, "SampleEnergy is not implemented"); };
    virtual void samplePosDir(Vector &pos, Vector &dir) override 
    { PROMPT_THROW(CalcError, "SamplePosDir is not implemented"); };
  private:
    PyObject *m_pyobj;
  };
}



#ifdef __cplusplus
extern "C" {
#endif

  void* pt_PythonGun_new(PyObject *pyobj);
  void pt_PythonGun_delete(void *obj);
  void pt_PythonGun_generate(void *obj);


#ifdef __cplusplus
}
#endif



#endif
