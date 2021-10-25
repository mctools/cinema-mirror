#include "PTAnaManager.hh"
#include "PTUtils.hh"

Prompt::AnaManager::AnaManager()
{}

Prompt::AnaManager::~AnaManager()
{}

void Prompt::AnaManager::addScorer(size_t id, const std::string &definition)
{
  std::vector<std::string> words;
  split(definition, ';', words);
}
