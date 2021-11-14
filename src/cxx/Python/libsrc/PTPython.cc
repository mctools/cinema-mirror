#include "PTPython.hh"
#include "PTLauncher.hh"

namespace pt = Prompt;

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
