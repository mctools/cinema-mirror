#ifndef Prompt_Utils_hh
#define Prompt_Utils_hh

#include "PromptCore.hh"

namespace Prompt {
  std::vector<std::string> split(const std::string& text, char delimiter);
  Vector string2vec(const std::string& text, char delimiter=',');
}

#endif
