#include "PTPython.hh"
#include "PTLauncher.hh"

namespace pt = Prompt;


double pt_rand_generate()
{
  return pt::Singleton<pt::SingletonPTRand>::getInstance().generate();  
}

void* pt_Launcher_getInstance()
{
  return static_cast<void *>(std::addressof(pt::Singleton<pt::Launcher>::getInstance()));
}

void pt_Launcher_setSeed(void* obj, uint64_t seed)
{
  static_cast<pt::Launcher *>(obj)->setSeed(seed);
}

void pt_Launcher_setGun(void* obj, void* objgun)
{

}

void pt_Launcher_loadGeometry(void* obj, const char* fileName)
{
  static_cast<pt::Launcher *>(obj)->loadGeometry(std::string(fileName));
}

size_t pt_Launcher_getTrajSize(void* obj)
{
  return static_cast<pt::Launcher *>(obj)->getTrajSize();
}

void pt_Launcher_getTrajectory(void* obj, double *trj)
{
  auto &trjv = static_cast<pt::Launcher *>(obj)->getTrajectory();
  for(const auto &v: trjv)
  {
    *(trj++) = v.x();
    *(trj++) = v.y();
    *(trj++) = v.z();
  }
}

void pt_Launcher_go(void* obj, uint64_t numParticle, double printPrecent, bool recordTrj)
{
  static_cast<pt::Launcher *>(obj)->go(numParticle, printPrecent, recordTrj);
}
