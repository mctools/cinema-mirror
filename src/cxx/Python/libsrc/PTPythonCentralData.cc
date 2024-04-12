
#include "PTPython.hh"
#include "PTCentralData.hh"

namespace pt = Prompt;

void* pt_CentralData_getInstance()
{
  return static_cast<void *>(std::addressof(pt::Singleton<pt::CentralData>::getInstance()));
}

double pt_CentralData_getGidiThreshold(void* obj)
{
    return static_cast<pt::CentralData *>(obj)->getGidiThreshold();
}

void pt_CentralData_setGidiThreshold(void* obj, double t)
{
    static_cast<pt::CentralData *>(obj)->setGidiThreshold(t);
}

const char * pt_CentralData_getGidiPops(void* obj)
{
    return static_cast<pt::CentralData *>(obj)->getGidiPops().c_str();
}

void pt_CentralData_setGidiPops(void* obj, const char *s)
{
    static_cast<pt::CentralData *>(obj)->setGidiPops(s);
}

const char * pt_CentralData_getGidiMap(void* obj)
{
    return static_cast<pt::CentralData *>(obj)->getGidiMap().c_str();
}

void pt_CentralData_setGidiMap(void* obj, const char *s)
{
    static_cast<pt::CentralData *>(obj)->setGidiMap(s);
}

bool pt_CentralData_getEnableGidi(void* obj)
{
    return static_cast<pt::CentralData *>(obj)->getEnableGidi();
}

void pt_CentralData_setEnableGidi(void* obj, bool t)
{
    return static_cast<pt::CentralData *>(obj)->setEnableGidi(t);
}


bool pt_CentralData_getEnableGidiPowerIteration(void* obj)
{
    return static_cast<pt::CentralData *>(obj)->getEnableGidiPowerIteration();
}

void pt_CentralData_setEnableGidiPowerIteration(void* obj, bool t)
{
    return static_cast<pt::CentralData *>(obj)->setEnableGidiPowerIteration(t);
}


