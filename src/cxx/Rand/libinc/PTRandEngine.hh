#ifndef Prompt_RandEngine_hh
#define Prompt_RandEngine_hh

#include "PromptCore.hh"

namespace Prompt {
  //This is the RandXRSR class of NCrystal to be removed

  class RandEngine final {
  public:
    RandEngine(uint64_t seed = 6402);
    double operator()();
    uint64_t min() const {return 0;}
    uint64_t max() const {return std::numeric_limits<uint64_t>::max();}
    ~RandEngine();
  private:
    void seed(uint64_t seed);

    uint64_t genUInt64();
    static uint64_t splitmix64(uint64_t& state);
    uint64_t m_s[2];
  };

}

#endif
