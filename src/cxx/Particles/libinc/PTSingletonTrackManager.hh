#ifndef Prompt_SingletonTrackManager_hh
#define Prompt_SingletonTrackManager_hh

#include "PTSingleton.hh"

namespace Prompt {

  class SingletonTrackManager {
  public:

  private:
    friend class Singleton<SingletonTrackManager>;
    SingletonTrackManager();
    ~SingletonTrackManager();
  };
}

#endif
