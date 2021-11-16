#ifndef Prompt_Launcher_hh
#define Prompt_Launcher_hh

#include "PromptCore.hh"
#include "PTSingleton.hh"
#include "PTPrimaryGun.hh"

namespace Prompt {
  class Launcher {
  public:
    void go(uint64_t numParticle, double printPrecent, bool recordTrj=false);
    void loadGeometry(const std::string &geofile);
    void setSeed(uint64_t seed) { Singleton<SingletonPTRand>::getInstance().setSeed(seed); }
    uint64_t getSeed() { return Singleton<SingletonPTRand>::getInstance().getSeed(); }
    void setGun(std::shared_ptr<PrimaryGun> gun) { m_gun=gun; }
    const std::vector<Vector> &getTrajectory() { return m_trajectory; }
    size_t getTrajSize() { return m_trajectory.size(); }

  private:
    friend class Singleton<Launcher>;
    Launcher();
    ~Launcher();
    std::shared_ptr<PrimaryGun> m_gun;
    std::vector<Vector> m_trajectory;
  };
}
#endif
