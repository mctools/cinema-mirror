#ifndef Prompt_SingletonTrackManager_hh
#define Prompt_SingletonTrackManager_hh

#include "PTSingleton.hh"
#include "PTTrack.hh"
#include "PromptCore.hh"

namespace Prompt {

  class TrackManager {
  public:
    void addTrack(std::unique_ptr<Prompt::Particle> &aparticle);

  private:
    friend class Singleton<TrackManager>;
    TrackManager();
    ~TrackManager();
    std::vector<std::unique_ptr<Particle> > m_tracks;
  };
}

#endif
