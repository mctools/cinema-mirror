#ifndef Prompt_ModelManager_hh
#define Prompt_ModelManager_hh

#include <string>
#include <memory>
#include "PromptCore.hh"
#include "PTVector.hh"

namespace Prompt {

  class ModelManager  {
  public:
    ModelManager();
    ~ModelManager();
  private:
    std::vector<std::shared_ptr<PhysicsModel> > m_models;
  };
}

#endif
