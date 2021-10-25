#ifndef Prompt_Utils_hh
#define Prompt_Utils_hh

#include "PromptCore.hh"

namespace Prompt {
  void split(const std::string& text, char delimiter, std::vector<std::string> &words);
}

#endif
