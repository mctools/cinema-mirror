#ifndef Prompt_RandCanonical_hh
#define Prompt_RandCanonical_hh

#include <functional>
#include <vector>
#include <memory>
#include <random>
#include <limits>

#include "PromptCore.hh"
#include "PTSingleton.hh"

//fixme: use ncrystal internal random generator
namespace Prompt {

  template <class T>
  class RandCanonical {
  public:
    RandCanonical(std::shared_ptr<T> gen);
    ~RandCanonical();
    double generate() const;

  private:
    uint64_t m_seed;
    std::shared_ptr<T> m_generator;
  };

  //fixme: can't set SEED !!!
  class SingletonPTRand : public RandCanonical<std::mt19937_64>  {
  private:
    friend class Singleton<SingletonPTRand>;
    SingletonPTRand(): RandCanonical<std::mt19937_64>(std::make_shared<std::mt19937_64>()) {}
    ~SingletonPTRand() {};
  };

}
#include "PTRandCanonical.tpp"


#endif
