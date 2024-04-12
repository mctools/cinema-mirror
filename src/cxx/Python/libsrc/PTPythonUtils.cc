

#include "PTPython.hh"
#include "PTPhysicsFactory.hh"

double pt_nccalNumDensity(const char *s)
{
    return Prompt::Singleton<Prompt::PhysicsFactory>::getInstance().nccalNumDensity(s);
}

