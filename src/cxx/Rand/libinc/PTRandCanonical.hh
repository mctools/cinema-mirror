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


  class PTRand : public RandCanonical<std::mt19937_64>  {
  public:
    PTRand(): RandCanonical<std::mt19937_64>(std::make_shared<std::mt19937_64>(6402)) {}
    ~PTRand() {};
  };

  using SingletonPTRand =  Singleton<PTRand> ;

}
#include "PTRandCanonical.tpp"


#endif
